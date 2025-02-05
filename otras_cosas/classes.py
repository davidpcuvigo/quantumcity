from graphviz import Source

# Load the DOT file
dot_file = "diagram.dot"

# Render the diagram as a PDF
src = Source.from_file(dot_file)
src.render("class_diagram", format="pdf", cleanup=True)
