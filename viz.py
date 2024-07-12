#%%
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager, rc
rc('font', family='AppleGothic', weight='bold')
#%%
years = [1970, 1980, 1990, 2000, 2010, 2020, 2023]

x_values = [1, 15, 172, 1280, 9770, 22400, 32200]

plt.figure(figsize=(10, 6))

plt.fill_between(years, x_values, color = "tab:blue")

plt.plot(years, x_values, color = "black", lw=2)

# plt.title('텍스트 요약 관련 논문 수', fontsize = 40)
# plt.xlabel('연도', fontsize = 30)
# plt.ylabel('논문 수',fontsize = 30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()

plt.tight_layout()
plt.show()

# %%
