from wordcloud import WordCloud,ImageColorGenerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.misc import imread

text = open('cut_words.txt', 'r').read()

bg_pic = imread('damo.png')
 
font = r'/mnt/c/Windows/Fonts/苹方黑体-中粗-简_0.ttf'

wc = WordCloud(mask=bg_pic, background_color='white', font_path=font, scale=1.5).generate(text)
image_colors = ImageColorGenerator(bg_pic)

plt.imshow(wc)
plt.axis('off')
plt.show()

wc.tofile('wc.jpg')