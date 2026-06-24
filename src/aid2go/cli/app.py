import typer
from aid2go.cli import preprocess_cmd, train_cmd, explain_cmd

app = typer.Typer(help="Aid2Go: Protein Function Prediction CLI")

app.add_typer(preprocess_cmd.app, name="preprocess", help="Data preprocessing commands")
app.add_typer(train_cmd.app, name="train", help="Model training commands")
app.add_typer(explain_cmd.app, name="explain", help="Explainability commands")

if __name__ == "__main__":
    app()
