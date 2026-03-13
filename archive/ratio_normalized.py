import matplotlib.pyplot as plt

# Data
bandwidths = [10, 8, 5, 3, 1, 0.8, 0.6, 0.4]  # Bandwidth levels in GB
  # CPU frequencies
operators = ["addmm_cpu", "bmm_cpu", "relu_cpu", "embedding_bag_cpu", 
             "bmm_backward_cpu", "addmm_backward_cpu", "relu_cpu_backward", 
             "embedding_bag_backward_cpu"]




# Example Ratios Data (replace this with actual values)
ratios_data = [
    # 10GB bandwidth
    [409/73, 108/73, 149/73, 1266/73, 1, 1497/73, 111/73, 734/73],
    # 8GB bandwidth
    [236/35, 48/35, 117/35, 947/35, 1, 1279/35, 11/5, 144/5],
    # 5GB bandwidth
    [20/3, 13/9, 121/36, 137/6, 1, 1451/36, 19/9, 157/6],
    # 3GB bandwidth
    [240/41, 57/41, 121/41, 847/41, 1, 1442/41, 77/41, 957/41],
    # 1GB bandwidth
    [236/69, 31/23, 124/69, 836/69, 1, 1663/69, 30/23, 913/69],
    # 0.8GB bandwidth
    [239/81, 112/81, 127/81, 259/27, 1, 1799/81, 97/81, 301/27],
    # 0.6GB bandwidth
    [241/98, 139/98, 135/98, 751/98, 1, 142/7, 54/49, 857/98],
    # 0.4GB bandwidth
    [253/127, 190/127, 162/127, 748/127, 130/127, 2320/127, 1, 814/127],
]
# ratios_data = [
#     # 10GB bandwidth
#     [9.408787669657235, 2.484472049689441, 3.4276512537382104, 29.123533471359558, 1.6793190706234185, 34.437543133195305, 2.5534851621808143, 16.885208189556018],
#     # 8GB bandwidth
#     [6.298372030958101, 1.281024819855885, 3.1224979983987193, 25.27355217507394, 0.9340805978115827, 34.13397384574326, 2.054977315185482, 26.901521216973585],
#     # 5GB bandwidth
#     [6.41711229946524, 1.3903743315508021, 3.2352941176470584, 21.978609625668447, 0.962566844919786, 38.79679144385027, 2.0320855614973263, 25.187165775401066],
#     # 3GB bandwidth
#     [6.345848757271284, 1.50713907895193, 3.199365415124272, 22.39555790586991, 1.0840824960338444, 38.1279741660497, 2.035959809624537, 25.304071919619247],
#     # 1GB bandwidth
#     [5.864811133200796, 2.3111332007952288, 3.081510934393638, 20.775347912524854, 1.714711729622265, 41.3270377333585, 2.2365805168986084, 22.68886679920477],
#     # 0.8GB bandwidth
#     [5.77992744860943, 2.708585247883917, 3.0713422007255136, 18.790810157194677, 1.9588875453446186, 43.506650544135425, 2.3458282950423213, 21.837968561064084],
#     # 0.6GB bandwidth
#     [5.582580495714617, 3.2198285846652768, 3.127171646977068, 17.396340050961317, 2.2700949733611306, 46.05049803104008, 2.5017373175816545, 19.851748899698865],
#     # 0.4GB bandwidth
#     [5.333052276559865, 4.0050502919224285, 3.414839797639123, 15.767284991568298, 2.74030354135346, 48.90387858347386, 2.6770657672849292, 17.15851602023609],
# ]

normalized_ratios = [
    [r / sum(ratio_set) * 100 for r in ratio_set]
    for ratio_set in ratios_data
]




total_times = [
    1582.49747133255,   # Total time for 2.4GHz
    1842.4020159244537,  # Total time for 2.0GHz
    2556.820482969284,   # Total time at 1.6GHz
    2537.4431874752045,  # Total time at 1.4GHz
    2928.8821828365326,  # Total time at 1GHz
    2872.862233877182,   # Total time at 0.6GHz
    3168.757362127304,    # Total time at 0.2GHz
    3090.1701748371124
    
]

# Convert data into a DataFrame for easier plotting
import pandas as pd
df_ratios = pd.DataFrame(normalized_ratios, columns=operators, index=bandwidths)

# Plotting
plt.figure(figsize=(12, 8))
for operator in operators:
    plt.plot(df_ratios.index, df_ratios[operator], marker='o', label=operator)

# Formatting
plt.title("CPU Percentages Ratios Across Operators (NUMA1)")
plt.xlabel("Memory Bandwidth in GB")  # Update x-axis label
plt.ylabel("Ratio Value")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
