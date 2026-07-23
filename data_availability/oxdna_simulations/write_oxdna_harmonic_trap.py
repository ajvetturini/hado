"""
This script shows how the oxDNA harmonic traps file was set to be used with the oxDNA simulation protocol]
"""
import plotly.graph_objs as go


SHOW = False
INPUT_DAT_PATH = r""  # Needed to get pos for harmonic traps
TENSION_OR_COMPRESSION = 'compression'
OUTPUT_FILENAME = f'{TENSION_OR_COMPRESSION}_external_forces.txt'

# These are the INDICES of the nucleotides to strain (and fix)
strained = [
    72668,72669,72670,72671,72672,72673,72674,72675,87230,87231,87232,87233,87234,87235,87236,87237,85629,85628,85627,85626,85625,85624,85623,85622,72694,72695,72696,72697,72698,72700,72699,72701,85880,85879,85878,85877,85876,85875,85874,85873,85349,85348,85347,85346,85345,85344,85343,85342,86487,86488,86489,86490,86491,86492,86493,86494,72667,72666,72665,72664,72663,72662,72661,72660,58017,58045,58046,58047,58048,58049,58050,58051,58052,70980,70979,70978,70977,70976,70975,70974,70973,72588,72587,72586,72585,72584,72583,72582,72581,58019,58020,58021,58022,58023,58024,58025,58026,58018,58016,58015,58014,58013,58012,58011,71838,71839,71840,71841,71842,71843,71844,71845,71231,71230,71229,71228,71227,71226,71225,71224,70693,70694,70695,70696,70697,70698,70699,70700
]
fixed = [
    25188,25187,25186,25185,25184,25183,25182,25181,14398,14399,14400,14401,14402,14403,14404,14405,14331,14332,14333,14334,14335,14336,14337,14338,26302,26301,26300,26299,26298,26297,26296,26295,26067,14351,14352,26066,26065,14353,14354,26064,26063,26062,26061,26060,14355,14356,14357,14358,24413,24412,24411,24410,24409,24408,24407,24406,14378,14379,14380,14381,14382,14383,14384,14385,40,41,42,43,44,45,46,47,11756,11755,11754,11753,11752,11751,11750,11749,10102,10101,10100,10099,67,68,69,70,71,72,73,74,10098,10097,10096,10095,10877,10876,10875,10874,10873,10872,10871,10870,87,88,89,90,91,92,93,94,20,21,22,23,24,25,26,27,11991,11990,11989,11988,11987,11986,11985,11984
]

k = 1.0  # Harmonic trap stiffness, 1.0 is fine for simple strain test I found
RATE = 1e-6  # Rate of compression. Too fast (e.g., 1e-4 or 1e-5) leads to metastable states


assert len(fixed) == len(strained), "Mismatched number of fixed and compressed nucleotides"

def read_dat(dat_path):
    positions = {}
    nt_idx = 0
    with open(dat_path, 'r') as f:
        for line in f:
            temp = line.split(' ')
            if len(temp) > 6:
                positions[nt_idx] = [float(temp[0]), float(temp[1]), float(temp[2])]
                nt_idx += 1
    return positions

# {
# type = trap
# particle = 2
# pos0 = 0., 0., 0.
# stiff = 1.0
# rate = 0.
# dir = 1.,0.,0.
# }

# My inputs are aligned along x axis so I just use dir = 1,0,0 for tension and -1,0,0 for compression
assert TENSION_OR_COMPRESSION in ['tension', 'compression'], "TENSION_OR_COMPRESSION must be 'tension' or 'compression'"
direction_strain = 1 if TENSION_OR_COMPRESSION == 'tension' else -1
direction_fixed = -direction_strain

data = read_dat(INPUT_DAT_PATH)
with open(OUTPUT_FILENAME, 'w') as f:
    for nt in fixed:
        f.write(f'{{\n')
        f.write(f'type = trap\n')
        f.write(f'particle = {nt}\n')
        f.write(f'pos0 = {data[nt][0]}, {data[nt][1]}, {data[nt][2]}\n')
        f.write(f'stiff = {k}\n')
        f.write(f'rate = 0\n')
        f.write(f'dir = {direction_strain}, 0, 0\n')
        f.write(f'}}\n\n')


    for nt in strained:
        f.write(f'{{\n')
        f.write(f'type = trap\n')
        f.write(f'particle = {nt}\n')
        f.write(f'pos0 = {data[nt][0]}, {data[nt][1]}, {data[nt][2]}\n')
        f.write(f'stiff = {k}\n')
        f.write(f'rate = {RATE}\n')
        f.write(f'dir = {direction_strain}, 0, 0\n')
        f.write(f'}}\n\n')


if SHOW:
    # Plot for verification
    x_vals, y_vals, z_vals = [], [], []
    colors = []
    sizes = []
    text_labels = []

    for idx, coords in data.items():
        x_vals.append(coords[0])
        y_vals.append(coords[1])
        z_vals.append(coords[2])

        if idx in fixed:
            colors.append('black')
            sizes.append(10)
            text_labels.append("")
        elif idx in strained:
            colors.append('green')
            sizes.append(10)
            text_labels.append("")
        else:
            colors.append('lightgrey')
            sizes.append(3)
            text_labels.append("")

    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers+text',  # Updated mode to include text
        text=text_labels,  # Assign the labels
        textposition="middle center",  # Center the text on the dot
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=0)
        ),
        textfont=dict(
            color='black',  # White text usually pops better on blue dots
            size=22
        )
    )])
    fig.update_layout(
        scene=dict(
            aspectmode='data'
        )
    )
    fig.show()
