import matplotlib.pyplot as plt
import pandas as pd

# Parsing the ratios
ratios_data = [
    "223/6:1:17/3:35/6:1:383/6:16/3:157/6",
    "77/2:7/6:35/6:41/6:1:389/6:16/3:28",
    "36:7/6:11/2:23/3:1:63:31/6:82/3",
    "204/5:6/5:6:52/5:1:347/5:28/5:154/5",
    "39:7/6:17/3:19/3:1:200/3:17/3:59/2",
    "116/3:7/6:6:7:1:129/2:35/6:83/3",
    "107/3:7/6:16/3:9:1:371/6:16/3:53/2",
    "223/6:7/6:17/3:65/6:1:127/2:17/3:27",
    "236/7:8/7:36/7:36/7:1:58:40/7:169/7",
    "71/2:4/3:16/3:23/3:1:123/2:35/6:26",
    "216/7:8/7:33/7:65/7:1:376/7:36/7:159/7",
    "221/7:8/7:5:83/7:1:386/7:38/7:163/7",
    "241/9:10/9:13/3:37/9:1:422/9:16/3:59/3",
    "80/3:10/9:38/9:58/9:1:421/9:49/9:172/9",
    "251/9:10/9:41/9:95/9:1:431/9:17/3:179/9",
    "117/4:5/4:37/8:117/8:1:415/8:47/8:179/8",
    "24:9/8:9/2:11/4:1:603/16:13/2:151/8",
    "363/16:19/16:17/4:99/16:1:303/8:101/16:75/4",
    "329/16:9/8:57/16:51/4:1:551/16:25/4:149/8",
    "114/5:6/5:62/15:256/15:1:592/15:20/3:98/5"
]

total_times = [
    585.1121509075165, 585.1505134105682, 584.2253134250641, 606.4811851978302,
    548.7877306938171, 586.6048376560211, 597.9654502868652, 584.9448008537292,
    582.9438662528992, 603.4069447517395, 600.3184628486633, 595.6758754253387,
    582.5849251747131, 589.7300179004669, 579.6998336315155, 591.372154712677,
    636.3770563602448, 627.0018672943115, 647.4523801803589, 637.2862422466278
]
embedding_sizes_mb = [0.48896, 4.8896, 24.41216, 48.896]* 5
# Define the operators in order
operators = [
    'addmm_cpu', 'bmm_cpu', 'relu_cpu', 'embedding_bag_cpu', 
    'bmm_backward_cpu', 'addmm_backward_cpu', 'relu_cpu_backward', 'embedding_bag_backward_cpu'
]

# Converting the string ratios into numerical values
ratios_values = []
index = 0
for ratio in ratios_data:
    if index>-1 and index <20:
        index+=1
        values = []
        for part in ratio.split(':'):
            num, denom = map(int, part.split('/')) if '/' in part else (int(part), 1)
            values.append(num / denom)
        ratios_values.append(values)

# Creating a DataFrame
df_ratios = pd.DataFrame(ratios_values, columns=operators)

# Plotting
plt.figure(figsize=(12, 8))
for operator in operators:
    plt.plot(df_ratios.index, df_ratios[operator], marker='o', label=operator)
    # plt.plot(embedding_sizes_mb[start:end+1], total_times[start:end+1], marker='o', color='black', linestyle='--', label='Total Time')
plt.xticks(ticks=range(len(df_ratios)), labels=embedding_sizes_mb, rotation=45)
plt.title("CPU Percentages Ratios Across Operators(NUMA1)")
plt.xlabel("Embedding Table Size (MB)")
plt.ylabel("Ratio Value")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()




# import matplotlib.pyplot as plt
# import pandas as pd

# # Define embedding table sizes in GB (expanded for five groups)
# embedding_sizes_gb = [0.0004775, 0.004775, 0.02384, 0.04775]

# # Parsing the ratios (added missing commas)
# ratios_data = [
#     "223/6:1:17/3:35/6:1:383/6:16/3:157/6",
#     "77/2:7/6:35/6:41/6:1:389/6:16/3:28",
#     "36:7/6:11/2:23/3:1:63:31/6:82/3",
#     "204/5:6/5:6:52/5:1:347/5:28/5:154/5",
#     "39:7/6:17/3:19/3:1:200/3:17/3:59/2",
#     "116/3:7/6:6:7:1:129/2:35/6:83/3",
#     "107/3:7/6:16/3:9:1:371/6:16/3:53/2",
#     "223/6:7/6:17/3:65/6:1:127/2:17/3:27",
#     "236/7:8/7:36/7:36/7:1:58:40/7:169/7",
#     "71/2:4/3:16/3:23/3:1:123/2:35/6:26",
#     "216/7:8/7:33/7:65/7:1:376/7:36/7:159/7",
#     "221/7:8/7:5:83/7:1:386/7:38/7:163/7",
#     "241/9:10/9:13/3:37/9:1:422/9:16/3:59/3",
#     "80/3:10/9:38/9:58/9:1:421/9:49/9:172/9",
#     "251/9:10/9:41/9:95/9:1:431/9:17/3:179/9",
#     "117/4:5/4:37/8:117/8:1:415/8:47/8:179/8",
#     "24:9/8:9/2:11/4:1:603/16:13/2:151/8",
#     "363/16:19/16:17/4:99/16:1:303/8:101/16:75/4",
#     "329/16:9/8:57/16:51/4:1:551/16:25/4:149/8",
#     "114/5:6/5:62/15:256/15:1:592/15:20/3:98/5"
# ]

# # Define the operators in order
# operators = [
#     'addmm_cpu', 'bmm_cpu', 'relu_cpu', 'embedding_bag_cpu', 
#     'bmm_backward_cpu', 'addmm_backward_cpu', 'relu_cpu_backward', 'embedding_bag_backward_cpu'
# ]

# # Converting the string ratios into numerical values
# ratios_values = []
# for ratio in ratios_data:
#     values = []
#     for part in ratio.split(':'):
#         num, denom = map(int, part.split('/')) if '/' in part else (int(part), 1)
#         values.append(num / denom)
#     ratios_values.append(values)

# # Creating a DataFrame
# df_ratios = pd.DataFrame(ratios_values, columns=operators)

# # Plotting grouped by index range
# plt.figure(figsize=(15, 12))
# group_ranges = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19)]
# titles = ["Bandwidth 20GB", "Bandwidth 8GB", "Bandwidth 5GB", "Bandwidth 3GB", "Bandwidth 1GB"]

# for i, (start, end) in enumerate(group_ranges):
#     plt.subplot(3, 2, i + 1)  # Adjusted to 3x2 layout for five subplots
#     for operator in operators:
#         plt.plot(embedding_sizes_gb, df_ratios[operator][start:end+1], marker='o', label=operator)
    
#     plt.title(f"Operator Ratios - {titles[i]}")
#     plt.xlabel("Embedding Table Size (GB)")
#     plt.ylabel("Ratio Value")
#     plt.legend(loc="upper left", fontsize='small')
#     plt.grid(True)

# plt.tight_layout()
# plt.show()
import matplotlib.pyplot as plt

# Define the embedding table sizes repeated five times
embedding_sizes_mb = [0.48896, 4.8896, 24.41216, 48.896] * 5

# Define the total times based on the values extracted from the user's data
total_times = [
    585.1121509075165, 585.1505134105682, 584.2253134250641, 606.4811851978302, 
    548.7877306938171, 586.6048376560211, 597.9654502868652, 584.9448008537292, 
    582.9438662528992, 603.4069447517395, 600.3184628486633, 595.6758754253387, 
    582.5849251747131, 589.7300179004669, 579.6998336315155, 591.372154712677, 
    636.3770563602448, 627.0018672943115, 647.4523801803589, 637.2862422466278
]
embedding_lookup_times = [
    1.582737922668457, 1.7473602294921875, 1.8988065719604492, 2.3041677474975586,
    1.716414451599121,1.8309314250946045, 2.243513822555542,  2.5387682914733887,
    1.5763466358184814, 2.0295698642730713, 2.65742564201355, 3.216601610183716, 1.5939583778381348,2.3504974842071533,3.4370994567871094,4.279906272888184,2.104022979736328,
    4.0292649269104, 8.20869779586792, 9.968750238418579
]

mlp_times = [
    8.770986795425415, 8.873723030090332, 8.38951849937439, 8.664812326431274, 8.919761657714844,9.052703619003296,8.634912729263306,8.707560777664185
    , 9.022839307785034, 8.784469366073608,
    8.85867714881897, 8.916109800338745, 9.362995147705078, 9.51225233078003,
    9.612519264221191, 9.286730527877808,17.08046269416809, 16.027968168258667, 15.138686418533325, 15.438634634017944
]



# Plotting
plt.figure(figsize=(10, 6))
plt.plot(total_times, marker='o', linestyle='-', color='b', label="Total Time")

plt.xticks(ticks=range(len(df_ratios)), labels=embedding_sizes_mb, rotation=45)
plt.xlabel("Embedding Table Size (MB)")
plt.ylabel("Total Time (s)")
plt.title("Total Time vs. Embedding Table Size (Repeated)")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(embedding_lookup_times, marker='o', linestyle='-', color='g', label="Embedding Forward Time")
plt.xticks(ticks=range(len(df_ratios)), labels=embedding_sizes_mb, rotation=45)
plt.xlabel("Embedding Table Size (MB)")
plt.ylabel("Embedding Forward Time (s)")
plt.title("Embedding Forward Time vs. Embedding Table Size (Repeated)")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(mlp_times, marker='o', linestyle='-', color='g', label="MLP Time")
plt.xticks(ticks=range(len(df_ratios)), labels=embedding_sizes_mb, rotation=45)
plt.xlabel("Embedding Table Size (MB)")
plt.ylabel("MLP Time (s)")
plt.title("MLP Time vs. Embedding Table Size (Repeated)")
plt.grid(True)
plt.legend()
plt.show()