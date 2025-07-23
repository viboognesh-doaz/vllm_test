import argparse
import os
template = '<!DOCTYPE html>\n<html>\n    <body>\n    <h1>Links for vLLM</h1/>\n        <a href="../{wheel_html_escaped}">{wheel}</a><br/>\n    </body>\n</html>\n'
parser = argparse.ArgumentParser()
parser.add_argument('--wheel', help='The wheel path.', required=True)
args = parser.parse_args()
filename = os.path.basename(args.wheel)
with open('index.html', 'w') as f:
    print(f'Generated index.html for {args.wheel}')
    f.write(template.format(wheel=filename, wheel_html_escaped=filename.replace('+', '%2B')))