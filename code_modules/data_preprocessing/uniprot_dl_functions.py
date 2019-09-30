import time
import re
import subprocess
from code_modules.data_preprocessing.preprocessing_functions import sspace


def download_uniprot(go_word, go_id, drop_go_term_col=False, do_call=True,
                     columns=None, taxonomy='Bacteria', len_min=0, len_max=100,
                     limit='', taxon_in_fname=False, add_taxon_col=False):
    # GO-TERM ###
    go_term_url = f"goa:(%22{sspace(go_word)}%20[{go_id}]%22)%20"

    # COLUMNS ###
    if not columns:
        columns = ['id', 'entry name', 'sequence', 'go(biological process)']

    if drop_go_term_col:
        del columns[columns.index('go(biological process)')]

    columns.append('lineage(ALL)') if add_taxon_col else None

    columns_url = ','.join([sspace(c) for c in columns])

    # TAXONOMY ###
    taxonomy_dict = {
        'Bacteria': 2,
        'Viridiplantae': 33090,
        'Bathycoccus prasinos': 41875,
        'Homo sapiens': 9606,
        'Pan': 9596,
        'Cercopithecidae': 9527,
        'Mus musculus': 10090,
        'Hymenoptera': 7399,
        'Viruses': 10239,
    }
    tax_url = sspace(f"%22{taxonomy} [{taxonomy_dict[taxonomy]}]")

    # LIMIT RESULTS ###
    limit_url = f"&limit={limit}" if limit else ''

    url = (
        f"https://www.uniprot.org/uniprot/?query={go_term_url}"
        f"taxonomy:{tax_url}%22%20length:[{len_min}%20TO%20{len_max}]"
        f"&format=tab&columns={columns_url}{limit_url}")

    timestamp = time.strftime("%d%m%y-%H%M")
    tax_fname = sspace(f'_{taxonomy}', '_') if taxon_in_fname else ''
    out_file = (
        f"/home/wogie/Downloads/uniprot_"
        f"{'_'.join(re.findall(re.compile('([a-zA-Z]+)'), go_term_url))}"
        f"{tax_fname}_{timestamp}.txt")

    command = ["wget", "-O", f"{out_file}", f"{url}"]

    print(f"Command: {' '.join(command)}")
    if do_call:
        subprocess.call(command)
