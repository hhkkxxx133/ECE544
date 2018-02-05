"""IO tools for mp0.
"""


def read_data_from_file(filename):
    """
    Read txt data from file.
    Each row is in the format article_id \t title \t positivity_score \n.
    Store this information in a python dictionary. Key: article_id(int),
    value: [title(str), score(float)].

    Args:
        filename(string): Location of the file to load.
    Returns:
        out_dict(dict): data loaded from file.
    """
    out_dict = {}
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            article_id, title, score = line.split('\t')
            out_dict[int(article_id)] = [title, float(score)]
    return out_dict


def write_data_to_file(filename, data):
    """
    Writes data to file in the format article_id\ttitle\tpositivity_score\n.

    Args:
        filename(string): Location of the file to save.
        data(dict): data for writting to file.
    """
    # pass
    with open(filename,'a') as f:
        for key, val in data.items():
            f.write(str(key)+'\t'+val[0]+'\t'+str(val[1])+'\n')


