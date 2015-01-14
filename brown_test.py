import numpy as np
from hmm import HMM

# open first 5 text files
# santize each one and add its contents (as characters) to a list

data = []
with open("A01", "r") as f:
    content = f.readlines()
    for line in content:
        # skip over the first 10 chars which are boiler and turn multiple whitespace into single
        line = line.replace('"', " ").replace(".", " ").replace(",", " ").replace(";", " ").replace(":", " ").replace("-"," ").replace("(", " ").replace(")", " ").lower()
        line = ' '.join(line[11:].split())
        for char in line:
            if (ord(char) >= 97 and ord(char) <= 122) or char == " ":
                data.append(char)
        data.append(" ")

obs_idx = 'abcdefghijklmnopqrstuvwxyz '
data = np.matrix(data)

marvin = HMM(2, 27, data, obs_idx)

marvin.train()
marvin.print_B()
