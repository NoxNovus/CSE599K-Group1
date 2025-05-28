import matplotlib.pyplot as plt

prefill = [0.018262537498958408, 0.018107026314828543, 0.01791859541554004,
           0.017978148174006493, 0.01801567799411714, 0.018005440221168102]
decode  = [0.47198368120007217, 0.9389015651890077, 1.8778181297006085,
           3.753182269714307, 7.530761553789489, 15.028420465497765]
overall = [0.49100119979120793, 0.9576715838978999, 1.8964377305936069,
           3.771925860235933, 7.549687713000457, 15.04755283978302]

x = list(range(5, 11))                       # log2(decode length) 5–10

plt.figure()
plt.plot(x, prefill, marker='o', label='Prefill')
plt.plot(x, decode,  marker='o', label='Decode')
plt.plot(x, overall, marker='o', label='Overall')

plt.xlabel('log₂(Decode Length)')
plt.ylabel('Time (seconds)')
plt.title('Prefill / Decode / Overall Time vs log₂(Decode Length)')
plt.xticks(x)
plt.grid(True)
plt.legend()
plt.show()
