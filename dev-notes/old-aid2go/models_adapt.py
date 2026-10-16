# Adaptative Sampling
class AdaptiveFocusedGATConvDU(GATConv):
    def __init__(self, *args, base_sample_size=8, max_sample_size=32, high_degree_threshold=10, directed=False, **kwargs):
        super(AdaptiveFocusedGATConvDU, self).__init__(*args, **kwargs)
        self.base_sample_size = base_sample_size
        self.max_sample_size = max_sample_size
        self.high_degree_threshold = high_degree_threshold
        self.directed = directed
        self.original_degrees = None

    def set_original_degrees(self, degrees):
        self.original_degrees = degrees

    def forward(self, x, edge_index, return_attention_weights=False):
        if self.original_degrees is None:
            raise ValueError("Original degrees have not been set. Call set_original_degrees() before forward pass.")

        row, col = edge_index
        
        # Compute adaptive sample sizes
        adaptive_sample_sizes = self._compute_adaptive_sample_sizes(row)
        
        # Create mask for high-degree nodes
        high_deg_mask = self.original_degrees > self.high_degree_threshold

        # Keep all edges for low-degree nodes
        low_deg_edges = edge_index[:, ~high_deg_mask[row]]

        # Sample edges for high-degree nodes
        high_deg_edges = []
        high_deg_nodes = high_deg_mask.nonzero(as_tuple=True)[0]
        
        for node in high_deg_nodes:
            if self.directed:
                # For directed graphs, consider only outgoing edges
                neighbors = row[col == node]
            else:
                # For undirected graphs, consider all connected nodes
                neighbors = torch.cat([col[row == node], row[col == node]])
            
            sample_size = adaptive_sample_sizes[node]
            if len(neighbors) > sample_size:
                sampled_neighbors = neighbors[torch.randperm(len(neighbors))[:sample_size]]
            else:
                sampled_neighbors = neighbors
            
            if self.directed:
                sampled_edges = torch.stack([torch.full_like(sampled_neighbors, node), sampled_neighbors], dim=0)
            else:
                sampled_edges = torch.stack([
                    torch.cat([torch.full_like(sampled_neighbors, node), sampled_neighbors]),
                    torch.cat([sampled_neighbors, torch.full_like(sampled_neighbors, node)])
                ], dim=0)
            
            high_deg_edges.append(sampled_edges)

        if high_deg_edges:
            high_deg_edges = torch.cat(high_deg_edges, dim=1)
            sampled_edges = torch.cat([low_deg_edges, high_deg_edges], dim=1)
        else:
            sampled_edges = low_deg_edges

        # Call the original forward method to compute attention
        out = super().forward(x, sampled_edges, return_attention_weights=True)
        x, (sampled_edges, alpha) = out

        if return_attention_weights:
            return x, (sampled_edges, alpha)
        else:
            return x

    def _compute_adaptive_sample_sizes(self, row):
        max_degree = self.original_degrees.max()
        adaptive_sample_sizes = torch.clamp(
            self.base_sample_size + (self.original_degrees - self.high_degree_threshold) * (self.max_sample_size - self.base_sample_size) / (max_degree - self.high_degree_threshold),
            self.base_sample_size,
            self.max_sample_size
        )
        return adaptive_sample_sizes.int()


class FocusedHGNNGATdirectAdapt(torch.nn.Module):
    def __init__(
        self,
        protein_features,
        go_features,
        out_features,
        dropout=0.6,
        heads=8,
        concat=True,
        protein_sample_size=8,
        protein_high_degree_threshold=10,
        go_base_sample_size=8,
        go_max_sample_size=32,
        go_high_degree_threshold=20,
    ):
        super(FocusedHGNNGATdirectAdapt, self).__init__()
        per_head_features = out_features // heads if concat else out_features

        self.conv1_protein = AdaptiveFocusedGATConvDU(
            protein_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            sample_size=protein_sample_size,
            high_degree_threshold=protein_high_degree_threshold,
            directed=False
        )
        self.conv1_go = AdaptiveFocusedGATConvDU(
            go_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            base_sample_size=go_base_sample_size,
            max_sample_size=go_max_sample_size,
            high_degree_threshold=go_high_degree_threshold,
            directed=True
        )
        self.conv2_protein = AdaptiveFocusedGATConvDU(
            out_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            sample_size=protein_sample_size,
            high_degree_threshold=protein_high_degree_threshold,
            directed=False
        )
        self.conv2_go = AdaptiveFocusedGATConvDU(
            out_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            base_sample_size=go_base_sample_size,
            max_sample_size=go_max_sample_size,
            high_degree_threshold=go_high_degree_threshold,
            directed=True
        )

        final_out_features = out_features if concat else per_head_features
        self.protein_to_go = Linear(final_out_features * 2, 1)

    def forward(self, x_dict, edge_index_dict, mask=None):
        # Calculate original degrees for GO nodes
        go_degrees = degree(edge_index_dict[("go", "relation", "go")][0], num_nodes=x_dict["go"].size(0))
        self.conv1_go.set_original_degrees(go_degrees)
        self.conv2_go.set_original_degrees(go_degrees)

        x_protein = F.elu(
            self.conv1_protein(
                x_dict["protein"], edge_index_dict[("protein", "ppi", "protein")]
            )
        )
        x_go = F.elu(
            self.conv1_go(x_dict["go"], edge_index_dict[("go", "relation", "go")])
        )

        x_protein = F.elu(
            self.conv2_protein(
                x_protein, edge_index_dict[("protein", "ppi", "protein")]
            )
        )
        x_go = F.elu(self.conv2_go(x_go, edge_index_dict[("go", "relation", "go")]))

        edge_index_protein_go = edge_index_dict[("protein", "associates_with", "go")]
        if mask is not None:
            edge_index_protein_go = edge_index_protein_go[:, mask]
        protein_features = x_protein[edge_index_protein_go[0]]
        go_features = x_go[edge_index_protein_go[1]]

        association_features = torch.cat([protein_features, go_features], dim=1)
        association_scores = self.protein_to_go(association_features).squeeze()
        return association_scores

    def get_attention_weights(self, x_dict, edge_index_dict):
        self.eval()
        with torch.no_grad():
            _, attention_protein_1 = self.conv1_protein(
                x_dict["protein"],
                edge_index_dict[("protein", "ppi", "protein")],
                return_attention_weights=True,
            )
            _, attention_protein_2 = self.conv2_protein(
                F.elu(self.conv1_protein(
                    x_dict["protein"], edge_index_dict[("protein", "ppi", "protein")])
                ),
                edge_index_dict[("protein", "ppi", "protein")],
                return_attention_weights=True,
            )

            _, attention_go_1 = self.conv1_go(
                x_dict["go"],
                edge_index_dict[("go", "relation", "go")],
                return_attention_weights=True,
            )
            _, attention_go_2 = self.conv2_go(
                F.elu(self.conv1_go(x_dict["go"], edge_index_dict[("go", "relation", "go")])),
                edge_index_dict[("go", "relation", "go")],
                return_attention_weights=True,
            )

            return [
                attention_protein_1,
                attention_go_1,
                attention_protein_2,
                attention_go_2,
            ]


################################################################

class AdaptiveFocusedGATConvProtein(GATConv):
    def __init__(self, *args, sample_size=8, high_degree_threshold=10, **kwargs):
        super(AdaptiveFocusedGATConvProtein, self).__init__(*args, **kwargs)
        self.sample_size = sample_size
        self.high_degree_threshold = high_degree_threshold
        self.original_degrees = None

    def set_original_degrees(self, degrees):
        self.original_degrees = degrees

    def forward(self, x, edge_index, return_attention_weights=False):
        if self.original_degrees is None:
            raise ValueError("Original degrees have not been set. Call set_original_degrees() before forward pass.")

        row, col = edge_index

        # Create mask for high-degree nodes
        high_deg_mask = self.original_degrees > self.high_degree_threshold

        # Keep all edges for low-degree nodes
        low_deg_edges = edge_index[:, ~high_deg_mask[row]]

        # Sample edges for high-degree nodes
        high_deg_edges = []
        high_deg_nodes = high_deg_mask.nonzero(as_tuple=True)[0]
        
        for node in high_deg_nodes:
            neighbors = torch.cat([row[col == node], col[row == node]])
            
            if len(neighbors) > self.sample_size:
                sampled_neighbors = neighbors[torch.randperm(len(neighbors))[:self.sample_size]]
            else:
                sampled_neighbors = neighbors
            
            sampled_edges = torch.stack([
                torch.cat([torch.full_like(sampled_neighbors, node), sampled_neighbors]),
                torch.cat([sampled_neighbors, torch.full_like(sampled_neighbors, node)])
            ], dim=0)
            
            high_deg_edges.append(sampled_edges)

        if high_deg_edges:
            high_deg_edges = torch.cat(high_deg_edges, dim=1)
            sampled_edges = torch.cat([low_deg_edges, high_deg_edges], dim=1)
        else:
            sampled_edges = low_deg_edges

        # Call the original forward method to compute attention
        out = super().forward(x, sampled_edges, return_attention_weights=True)
        x, (sampled_edges, alpha) = out

        if return_attention_weights:
            return x, (sampled_edges, alpha)
        else:
            return x

class AdaptiveFocusedGATConvGO(GATConv):
    def __init__(self, *args, base_sample_size=8, max_sample_size=32, high_degree_threshold=20, **kwargs):
        super(AdaptiveFocusedGATConvGO, self).__init__(*args, **kwargs)
        self.base_sample_size = base_sample_size
        self.max_sample_size = max_sample_size
        self.high_degree_threshold = high_degree_threshold
        self.original_degrees = None

    def set_original_degrees(self, degrees):
        self.original_degrees = degrees

    def forward(self, x, edge_index, return_attention_weights=False):
        if self.original_degrees is None:
            raise ValueError("Original degrees have not been set. Call set_original_degrees() before forward pass.")

        row, col = edge_index
        
        # Compute adaptive sample sizes
        adaptive_sample_sizes = self._compute_adaptive_sample_sizes()
        
        # Create mask for high-degree nodes
        high_deg_mask = self.original_degrees > self.high_degree_threshold

        # Keep all edges for low-degree nodes
        low_deg_edges = edge_index[:, ~high_deg_mask[row]]

        # Sample edges for high-degree nodes
        high_deg_edges = []
        high_deg_nodes = high_deg_mask.nonzero(as_tuple=True)[0]
        
        for node in high_deg_nodes:
            neighbors = row[col == node]
            
            sample_size = adaptive_sample_sizes[node]
            if len(neighbors) > sample_size:
                sampled_neighbors = neighbors[torch.randperm(len(neighbors))[:sample_size]]
            else:
                sampled_neighbors = neighbors
            
            sampled_edges = torch.stack([torch.full_like(sampled_neighbors, node), sampled_neighbors], dim=0)
            high_deg_edges.append(sampled_edges)

        if high_deg_edges:
            high_deg_edges = torch.cat(high_deg_edges, dim=1)
            sampled_edges = torch.cat([low_deg_edges, high_deg_edges], dim=1)
        else:
            sampled_edges = low_deg_edges

        # Call the original forward method to compute attention
        out = super().forward(x, sampled_edges, return_attention_weights=True)
        x, (sampled_edges, alpha) = out

        if return_attention_weights:
            return x, (sampled_edges, alpha)
        else:
            return x

    def _compute_adaptive_sample_sizes(self):
        max_degree = self.original_degrees.max()
        adaptive_sample_sizes = torch.clamp(
            self.base_sample_size + (self.original_degrees - self.high_degree_threshold) * (self.max_sample_size - self.base_sample_size) / (max_degree - self.high_degree_threshold),
            self.base_sample_size,
            self.max_sample_size
        )
        return adaptive_sample_sizes.int()

class FocusAdaptModel(torch.nn.Module):
    def __init__(
        self,
        protein_features,
        go_features,
        out_features,
        dropout=0.6,
        heads=8,
        concat=True,
        protein_sample_size=8,
        protein_high_degree_threshold=10,
        go_base_sample_size=8,
        go_max_sample_size=32,
        go_high_degree_threshold=20,
    ):
        super(FocusAdaptModel, self).__init__()
        per_head_features = out_features // heads if concat else out_features

        self.conv1_protein = AdaptiveFocusedGATConvProtein(
            protein_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            sample_size=protein_sample_size,
            high_degree_threshold=protein_high_degree_threshold
        )
        self.conv1_go = AdaptiveFocusedGATConvGO(
            go_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            base_sample_size=go_base_sample_size,
            max_sample_size=go_max_sample_size,
            high_degree_threshold=go_high_degree_threshold
        )
        self.conv2_protein = AdaptiveFocusedGATConvProtein(
            out_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            sample_size=protein_sample_size,
            high_degree_threshold=protein_high_degree_threshold
        )
        self.conv2_go = AdaptiveFocusedGATConvGO(
            out_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            base_sample_size=go_base_sample_size,
            max_sample_size=go_max_sample_size,
            high_degree_threshold=go_high_degree_threshold
        )

        final_out_features = out_features if concat else per_head_features
        self.protein_to_go = Linear(final_out_features * 2, 1)

    def forward(self, x_dict, edge_index_dict, mask=None):
        # Calculate original degrees for protein and GO nodes
        protein_degrees = degree(edge_index_dict[("protein", "ppi", "protein")][0], num_nodes=x_dict["protein"].size(0))
        go_degrees = degree(edge_index_dict[("go", "relation", "go")][0], num_nodes=x_dict["go"].size(0))
        
        self.conv1_protein.set_original_degrees(protein_degrees)
        self.conv2_protein.set_original_degrees(protein_degrees)
        self.conv1_go.set_original_degrees(go_degrees)
        self.conv2_go.set_original_degrees(go_degrees)

        x_protein = F.elu(
            self.conv1_protein(
                x_dict["protein"], edge_index_dict[("protein", "ppi", "protein")]
            )
        )
        x_go = F.elu(
            self.conv1_go(x_dict["go"], edge_index_dict[("go", "relation", "go")])
        )

        x_protein = F.elu(
            self.conv2_protein(
                x_protein, edge_index_dict[("protein", "ppi", "protein")]
            )
        )
        x_go = F.elu(self.conv2_go(x_go, edge_index_dict[("go", "relation", "go")]))

        edge_index_protein_go = edge_index_dict[("protein", "associates_with", "go")]
        if mask is not None:
            edge_index_protein_go = edge_index_protein_go[:, mask]
        protein_features = x_protein[edge_index_protein_go[0]]
        go_features = x_go[edge_index_protein_go[1]]

        association_features = torch.cat([protein_features, go_features], dim=1)
        association_scores = self.protein_to_go(association_features).squeeze()
        return association_scores

    def get_attention_weights(self, x_dict, edge_index_dict):
        self.eval()
        with torch.no_grad():
            _, attention_protein_1 = self.conv1_protein(
                x_dict["protein"],
                edge_index_dict[("protein", "ppi", "protein")],
                return_attention_weights=True,
            )
            _, attention_protein_2 = self.conv2_protein(
                F.elu(self.conv1_protein(
                    x_dict["protein"], edge_index_dict[("protein", "ppi", "protein")])
                ),
                edge_index_dict[("protein", "ppi", "protein")],
                return_attention_weights=True,
            )

            _, attention_go_1 = self.conv1_go(
                x_dict["go"],
                edge_index_dict[("go", "relation", "go")],
                return_attention_weights=True,
            )
            _, attention_go_2 = self.conv2_go(
                F.elu(self.conv1_go(x_dict["go"], edge_index_dict[("go", "relation", "go")])),
                edge_index_dict[("go", "relation", "go")],
                return_attention_weights=True,
            )

            return [
                attention_protein_1,
                attention_go_1,
                attention_protein_2,
                attention_go_2,
            ]
            
################################################################
def dynamic_sample_size(node_degree, base_size, max_size):
    return min(max(int(node_degree * 0.5), base_size), max_size)

class AFGATConvProtein(GATConv):
    def __init__(self, *args, base_sample_size=32, max_sample_size=128, high_degree_threshold=30, **kwargs):
        super(AFGATConvProtein, self).__init__(*args, **kwargs)
        self.base_sample_size = base_sample_size
        self.max_sample_size = max_sample_size
        self.high_degree_threshold = high_degree_threshold
        self.original_degrees = None

    def set_original_degrees(self, degrees):
        self.original_degrees = degrees

    def forward(self, x, edge_index, return_attention_weights=False):
        if self.original_degrees is None:
            raise ValueError("Original degrees have not been set. Call set_original_degrees() before forward pass.")

        row, col = edge_index

        # Create mask for high-degree nodes
        high_deg_mask = self.original_degrees > self.high_degree_threshold

        # Keep all edges for low-degree nodes
        low_deg_edges = edge_index[:, ~high_deg_mask[row]]

        # Sample edges for high-degree nodes
        high_deg_edges = []
        high_deg_nodes = high_deg_mask.nonzero(as_tuple=True)[0]
        
        for node in high_deg_nodes:
            neighbors = torch.cat([row[col == node], col[row == node]])
            
            sample_size = dynamic_sample_size(self.original_degrees[node], self.base_sample_size, self.max_sample_size)
            
            if len(neighbors) > sample_size:
                sampled_neighbors = neighbors[torch.randperm(len(neighbors))[:sample_size]]
            else:
                sampled_neighbors = neighbors
            
            sampled_edges = torch.stack([
                torch.cat([torch.full_like(sampled_neighbors, node), sampled_neighbors]),
                torch.cat([sampled_neighbors, torch.full_like(sampled_neighbors, node)])
            ], dim=0)
            
            high_deg_edges.append(sampled_edges)

        if high_deg_edges:
            high_deg_edges = torch.cat(high_deg_edges, dim=1)
            sampled_edges = torch.cat([low_deg_edges, high_deg_edges], dim=1)
        else:
            sampled_edges = low_deg_edges

        # Call the original forward method to compute attention
        out = super().forward(x, sampled_edges, return_attention_weights=True)
        x, (sampled_edges, alpha) = out

        if return_attention_weights:
            return x, (sampled_edges, alpha)
        else:
            return x

class AFGATConvGO(GATConv):
    def __init__(self, *args, base_sample_size=32, max_sample_size=128, high_degree_threshold=30, **kwargs):
        super(AFGATConvGO, self).__init__(*args, **kwargs)
        self.base_sample_size = base_sample_size
        self.max_sample_size = max_sample_size
        self.high_degree_threshold = high_degree_threshold
        self.original_degrees = None

    def set_original_degrees(self, degrees):
        self.original_degrees = degrees

    def forward(self, x, edge_index, return_attention_weights=False):
        if self.original_degrees is None:
            raise ValueError("Original degrees have not been set. Call set_original_degrees() before forward pass.")

        row, col = edge_index
        
        # Create mask for high-degree nodes
        high_deg_mask = self.original_degrees > self.high_degree_threshold

        # Keep all edges for low-degree nodes
        low_deg_edges = edge_index[:, ~high_deg_mask[row]]

        # Sample edges for high-degree nodes
        high_deg_edges = []
        high_deg_nodes = high_deg_mask.nonzero(as_tuple=True)[0]
        
        for node in high_deg_nodes:
            # Include both incoming and outgoing neighbors
            incoming_neighbors = row[col == node]
            outgoing_neighbors = col[row == node]
            neighbors = torch.cat([incoming_neighbors, outgoing_neighbors])
            
            sample_size = dynamic_sample_size(self.original_degrees[node], self.base_sample_size, self.max_sample_size)
            
            if len(neighbors) > sample_size:
                sampled_neighbors = neighbors[torch.randperm(len(neighbors))[:sample_size]]
            else:
                sampled_neighbors = neighbors
            
            # Create edges in both directions
            sampled_edges = torch.stack([
                torch.cat([torch.full_like(sampled_neighbors, node), sampled_neighbors]),
                torch.cat([sampled_neighbors, torch.full_like(sampled_neighbors, node)])
            ], dim=0)
            
            high_deg_edges.append(sampled_edges)

        if high_deg_edges:
            high_deg_edges = torch.cat(high_deg_edges, dim=1)
            sampled_edges = torch.cat([low_deg_edges, high_deg_edges], dim=1)
        else:
            sampled_edges = low_deg_edges

        # Call the original forward method to compute attention
        out = super().forward(x, sampled_edges, return_attention_weights=True)
        x, (sampled_edges, alpha) = out

        if return_attention_weights:
            return x, (sampled_edges, alpha)
        else:
            return x


class FocusAdaptModel2(torch.nn.Module):
    def __init__(
        self,
        protein_features,
        go_features,
        out_features,
        dropout=0.6,
        heads=8,
        concat=True,
        protein_base_sample_size=32,
        protein_max_sample_size=128,
        protein_high_degree_threshold=30,
        go_base_sample_size=32,
        go_max_sample_size=128,
        go_high_degree_threshold=30,
    ):
        super(FocusAdaptModel2, self).__init__()
        per_head_features = out_features // heads if concat else out_features

        self.conv1_protein = AFGATConvProtein(
            protein_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            base_sample_size=protein_base_sample_size,
            max_sample_size=protein_max_sample_size,
            high_degree_threshold=protein_high_degree_threshold
        )
        self.conv1_go = AFGATConvGO(
            go_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            base_sample_size=go_base_sample_size,
            max_sample_size=go_max_sample_size,
            high_degree_threshold=go_high_degree_threshold
        )
        self.conv2_protein = AFGATConvProtein(
            out_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            base_sample_size=protein_base_sample_size,
            max_sample_size=protein_max_sample_size,
            high_degree_threshold=protein_high_degree_threshold
        )
        self.conv2_go = AFGATConvGO(
            out_features,
            per_head_features,
            heads=heads,
            concat=concat,
            dropout=dropout,
            base_sample_size=go_base_sample_size,
            max_sample_size=go_max_sample_size,
            high_degree_threshold=go_high_degree_threshold
        )

        final_out_features = out_features if concat else per_head_features
        self.protein_to_go = Linear(final_out_features * 2, 1)

    def forward(self, x_dict, edge_index_dict, mask=None):
        # Calculate original degrees for protein and GO nodes
        protein_degrees = degree(edge_index_dict[("protein", "ppi", "protein")][0], num_nodes=x_dict["protein"].size(0))
        go_degrees = degree(edge_index_dict[("go", "relation", "go")][0], num_nodes=x_dict["go"].size(0))
        
        self.conv1_protein.set_original_degrees(protein_degrees)
        self.conv2_protein.set_original_degrees(protein_degrees)
        self.conv1_go.set_original_degrees(go_degrees)
        self.conv2_go.set_original_degrees(go_degrees)

        x_protein = F.elu(
            self.conv1_protein(
                x_dict["protein"], edge_index_dict[("protein", "ppi", "protein")]
            )
        )
        x_go = F.elu(
            self.conv1_go(x_dict["go"], edge_index_dict[("go", "relation", "go")])
        )

        x_protein = F.elu(
            self.conv2_protein(
                x_protein, edge_index_dict[("protein", "ppi", "protein")]
            )
        )
        x_go = F.elu(self.conv2_go(x_go, edge_index_dict[("go", "relation", "go")]))

        edge_index_protein_go = edge_index_dict[("protein", "associates_with", "go")]
        if mask is not None:
            edge_index_protein_go = edge_index_protein_go[:, mask]
        protein_features = x_protein[edge_index_protein_go[0]]
        go_features = x_go[edge_index_protein_go[1]]

        association_features = torch.cat([protein_features, go_features], dim=1)
        association_scores = self.protein_to_go(association_features).squeeze()
        return association_scores

    def get_attention_weights(self, x_dict, edge_index_dict):
        self.eval()
        with torch.no_grad():
            _, attention_protein_1 = self.conv1_protein(
                x_dict["protein"],
                edge_index_dict[("protein", "ppi", "protein")],
                return_attention_weights=True,
            )
            _, attention_protein_2 = self.conv2_protein(
                F.elu(self.conv1_protein(
                    x_dict["protein"], edge_index_dict[("protein", "ppi", "protein")])
                ),
                edge_index_dict[("protein", "ppi", "protein")],
                return_attention_weights=True,
            )

            _, attention_go_1 = self.conv1_go(
                x_dict["go"],
                edge_index_dict[("go", "relation", "go")],
                return_attention_weights=True,
            )
            _, attention_go_2 = self.conv2_go(
                F.elu(self.conv1_go(x_dict["go"], edge_index_dict[("go", "relation", "go")])),
                edge_index_dict[("go", "relation", "go")],
                return_attention_weights=True,
            )

            return [
                attention_protein_1,
                attention_go_1,
                attention_protein_2,
                attention_go_2,
            ]

################################################################


# Training loop
def train(
    model,
    dataset,
    config,
    run,
    save_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_scheduler=True,
    accumulation_steps=1,
):

    model.to(device)
    dataset.to_device(device)
    dataset.create_data_loaders(batch_size=config["batch_size"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            patience=config["scheduler_patience"],
            factor=0.1,
            verbose=True,
        )
        if use_scheduler
        else None
    )

    best_val_loss = float("inf")
    early_stopping_counter = 0

    for epoch in tqdm(
        range(1, config["epochs"] + 1), desc="Training...", total=config["epochs"]
    ):
        model.train()
        train_loss = 0
        train_predictions, train_labels = [], []

        for i, (edge_index, edge_label) in enumerate(dataset.train_loader):
            edge_index, edge_label = edge_index.to(device), edge_label.to(device)
            out = model(
                {
                    "protein": dataset.hetero_data["protein"].x,
                    "go": dataset.hetero_data["go"].x,
                },
                {
                    ("protein", "ppi", "protein"): dataset.hetero_data[
                        "protein", "ppi", "protein"
                    ].edge_index,
                    ("go", "relation", "go"): dataset.hetero_data[
                        "go", "relation", "go"
                    ].edge_index,
                    ("protein", "associates_with", "go"): edge_index.t(),
                },
            )
            loss = (
                F.binary_cross_entropy_with_logits(out, edge_label) / accumulation_steps
            )
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            train_predictions.append(out.sigmoid().detach().cpu().numpy())
            train_labels.append(edge_label.cpu().numpy())

        train_loss /= len(dataset.train_loader)
        train_auroc = roc_auc_score(
            np.concatenate(train_labels), np.concatenate(train_predictions)
        )
        wandb.log({"train_loss": train_loss, "train_auroc": train_auroc})

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_predictions, val_labels = [], []

            for edge_index, edge_label in dataset.val_loader:
                edge_index, edge_label = edge_index.to(device), edge_label.to(device)
                out = model(
                    {
                        "protein": dataset.hetero_data["protein"].x,
                        "go": dataset.hetero_data["go"].x,
                    },
                    {
                        ("protein", "ppi", "protein"): dataset.hetero_data[
                            "protein", "ppi", "protein"
                        ].edge_index,
                        ("go", "relation", "go"): dataset.hetero_data[
                            "go", "relation", "go"
                        ].edge_index,
                        ("protein", "associates_with", "go"): edge_index.t(),
                    },
                )
                loss = F.binary_cross_entropy_with_logits(out, edge_label)
                val_loss += loss.item()
                val_predictions.append(out.sigmoid().cpu().numpy())
                val_labels.append(edge_label.cpu().numpy())

            val_loss /= len(dataset.val_loader)
            val_auroc = roc_auc_score(
                np.concatenate(val_labels), np.concatenate(val_predictions)
            )
            wandb.log({"val_loss": val_loss, "val_auroc": val_auroc})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{save_path}/best_model_weights.pt")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        if scheduler:
            scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.3f}, Train AUROC: {train_auroc:.3f}, "
                f"Val Loss: {val_loss:.3f}, Val AUROC: {val_auroc:.3f}"
            )
    return model, best_val_loss
