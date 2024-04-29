
from pylatex import Document, Section, Subsection, Command, Figure, Tabular, Itemize
from pylatex.utils import italic, NoEscape, bold
from icecream import ic
import os

__all__ = ['generate_report']

def generate_report(report_info):
    report = Document('./output/report')
    report.preamble.append(Command('title', 'Simulation report'))
    report.preamble.append(Command('author','UVigo'))
    report.preamble.append(Command('date', NoEscape(r'\today')))
    report.append(NoEscape(r'\maketitle'))

    #Routing Phase section
    with report.create(Section('Routing Protocol')):
        report.append('The next section details information gathered through the route calculation phase\n')

        with report.create(Subsection('Network')):
            with report.create(Figure(position='ht!')) as fig_network:
                image_file = os.path.join(os.path.dirname(__file__), 'output/graf.png')
                fig_network.add_image(image_file,width='180px')
                fig_network.add_caption('Simulated network')

        with report.create(Subsection('Link fidelities')):
            report.append('Fidelity and Cost of each of the links\n')
            with report.create(Tabular('l|l|l|l')) as table:
                table.add_hline()
                table.add_row(('link','cost','fidelity','number of rounds'))
                table.add_hline()
                for key, value in report_info['link_fidelities'].items():
                    table.add_row((key,value[0],value[1],value[2]))
        with report.create(Subsection('Path simulation')):                
            #report.append('Request fulfillment\n')
            with report.create(Tabular('l|l|l|l|l')) as table:      
                table.add_hline()
                table.add_row(['request','result','fidelity','time','purification rounds'])
                table.add_hline()
                for reg in report_info['requests_status']:
                    table.add_row(bold(reg['request']),reg['result'],reg['fidelity'],reg['time'],reg['purif_rounds'])
                table.add_hline()
            report.append('\n')
            report.append('\n')
            with report.create(Tabular('l|p{3.75in}')) as table:
                table.add_hline()
                table.add_row('request','shortest path')
                table.add_hline()
                for reg in report_info['requests_status']:
                    table.add_row(bold(reg['request']),reg['shortest_path'])

    with report.create(Section('Fase de simulación')):
        report.append('Esta parte está pendiente')

    report.generate_pdf('./output/report',clean_tex=False)
    report.generate_tex()
