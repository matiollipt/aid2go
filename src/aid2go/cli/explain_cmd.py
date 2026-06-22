import typer
app = typer.Typer()

@app.command()
def explain():
    typer.echo("Explainability not yet implemented.")
