"""Convert HTK lattice file to dot graph."""
import os
import gzip
from subprocess import call
import argparse

def convert_lattice(file_in, file_out):
    """Convert lattice format to graphviz format."""
    open_fn = gzip.open if file_in.endswith('.gz') else open
    with open_fn(file_in, 'rt') as lattice, open(file_out, 'w') as dot:
        dot.write(
            "digraph lattice {\n" \
            "\trankdir=LR;\n" \
            "\tnode [shape = ellipse; fontname = courier];\n" \
            "\tedge [fontname = courier];\n\n")
        while True:
            line = lattice.readline()
            if line.startswith('N='):
                break
        first_line = line.split()
        nodes, links = [int(i.split('=')[1]) for i in first_line]
        for _ in range(nodes):
            next_line = lattice.readline().split()
            content = tuple(i.split('=')[1] for i in next_line[0:3])
            dot.write("\t%s [label = \"id=%s\\nt=%s\\nW=%s\"];\n" % (
                content[0], content[0], content[1], content[2]))
        dot.write("\n")
        for _ in range(links):
            next_line = lattice.readline().split()
            content = tuple(i.split('=')[1] for i in next_line[0:5])
            if next_line[5].startswith('n='):
                dot.write(
                    "\t%s -> %s [label = \"id=%s\\na=%s\\nl=%s\\nn=%s\"];\n" % (
                        content[1], content[2], content[0], content[3],
                        content[4], next_line[5].split('=')[1]))
            else:
                dot.write("\t%s -> %s [label = \"id=%s\\na=%s\\nl=%s\"];\n" % (
                    content[1], content[2], content[0], content[3], content[4]))
        dot.write("}")

def convert_confnet(file_in, file_out):
    """Convert confusion network format to graphviz format."""
    open_fn = gzip.open if file_in.endswith('.gz') else open
    with open_fn(file_in, 'rt') as confnet, open(file_out, 'w') as dot:
        dot.write(
            "digraph lattice {\n" \
            "\trankdir=LR;\n" \
            "\tnode [shape = ellipse; fontname = courier];\n" \
            "\tedge [fontname = courier];\n\n")
        confset = int(confnet.readline().strip().split('=')[1])
        for i in range(confset + 1):
            dot.write("\t%i [label = \"t=\"];\n" %i)
        for i in range(confset):
            num_arcs = int(confnet.readline().strip().split('=')[1])
            for _ in range(num_arcs):
                next_line = confnet.readline().split()
                word, prob = [next_line[k].split('=')[1] for k in [0, 3]]
                dot.write("\t%i -> %i [label = \"W=%s\\np=%s\"];\n" %(
                    i + 1, i, word, prob))
        dot.write("}")

def dot_2_pdf(dot_file, pdf_file):
    """Generate pdf file based on the graphviz file."""
    call(['dot', '-Tpdf', dot_file, '-o', pdf_file])

def main():
    """Main function for conversion."""
    parser = argparse.ArgumentParser(description="lattice visualisation")
    parser.add_argument('file_type', type=str, choices=['confnet', 'lattice'],
                        help="type of file to convert")
    parser.add_argument('file_in', type=str, help="path to input file")
    parser.add_argument('dir_out', type=str, default=None,
                        help="path to output dir")
    parser.add_argument('suffix', type=str, default='',
                        help="suffix to output file name")
    args = parser.parse_args()

    file_in = os.path.abspath(args.file_in)
    if not args.dir_out:
        file_name = os.path.splitext(file_in)[0]
        gv_file = file_name + args.suffix + '.gv'
        pdf_file = file_name + args.suffix + '.pdf'
    else:
        file_name = os.path.basename(file_in).split('.')[0]
        gv_file = os.path.join(args.dir_out, file_name + args.suffix + '.gv')
        pdf_file = os.path.join(args.dir_out, file_name + args.suffix + '.pdf')

    if args.file_type == 'lattice':
        convert_lattice(file_in, gv_file)
    elif args.file_type == 'confnet':
        convert_confnet(file_in, gv_file)
    else:
        raise NotImplementedError
    dot_2_pdf(gv_file, pdf_file)

if __name__ == "__main__":
    main()
