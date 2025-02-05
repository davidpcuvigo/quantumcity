from PyPDF2 import PdfMerger

merger = PdfMerger()
merger.append("/home/dperez/VirtualNetSquid/NetSquid/QuantumNetwork/graficosTFM/fidelity_demand.pdf")
merger.append("/home/dperez/VirtualNetSquid/NetSquid/QuantumNetwork/graficosTFM/queue_demand.pdf")
merger.write("combined.pdf")
merger.close()