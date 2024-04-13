
from pylatex import Document, Section, Subsection, Command, Figure, Tabular, Itemize
from pylatex.utils import italic, NoEscape
from icecream import ic

__all__ = ['generate_report','prueba']

def prueba():
    print('Hola')

def generate_report(report_info):
    report = Document()
    report.preamble.append(Command('title', 'Simulation report'))
    report.preamble.append(Command('author','UVigo'))
    report.preamble.append(Command('date', NoEscape(r'\today')))
    report.append(NoEscape(r'\maketitle'))

    #Routing Phase section
    with report.create(Section('Routing Protocol')):
        report.append('The next section details information gathered through the route calculation phase\n')
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
            for request_result in report_info['requests_status']:
                report.append(f"Request:{request_result['request']}")
                with report.create(Itemize()) as itemize:
                    itemize.add_item(f"Result: {request_result['result']}")
                    itemize.add_item(f"Sortest path: {request_result['shortest_path']}")
                    itemize.add_item(f"Fidelity: {request_result['fidelity']}")
                    itemize.add_item(f"Time: {request_result['time']}")
                    itemize.add_item(f"Purification rounds:{request_result['purif_rounds']}")

    with report.create(Section('Fase de simulación')):
        report.append('Esta parte está pendiente')

    report.generate_pdf('./output/report',clean_tex=False)
    report.generate_tex()
