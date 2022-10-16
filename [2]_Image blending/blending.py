import cv2

# IMAGE: Double Exposion
# đọc ảnh foreground
fg = cv2.imread(r'2.png')
print('Kich thuoc theo tung kenh cua foreground: ', fg.shape)

# đọc ảnh effect
eff = cv2.imread(r'3.jpg')
print('Kich thuoc theo tung kenh cua eff: ', eff.shape)

# đọc ảnh mask
mask = cv2.imread(r'1.png', cv2.IMREAD_UNCHANGED)
print('Kich thuoc theo tung kenh cua mask: ', mask.shape)


scale_percent = 40 # percent of original size
width = int(fg.shape[1] * scale_percent / 100)
height = int(fg.shape[0] * scale_percent / 100)
dim = (width, height)

fg = cv2.resize(fg, dim)
eff = cv2.resize(eff, dim)
mask = cv2.resize(mask, dim)

# ảnh chuẩn hóa = cv2.resize(ảnh gốc, (chiều rộng, chiều cao))

# Sao chép ảnh qua biến mới
result = fg.copy()
alpha = 0.9
for x in range(mask.shape[0]): # result.shape[0]: chiều cao ảnh
    for y in range(mask.shape[1]): # result.shape[1]: chiều rộng ảnh
        if (mask[x,y,3] != 0): # Kiểm tra điểm ảnh
            result[x,y] = (alpha * fg[x,y] + (1 - alpha) * eff[x,y])

cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()



# result = fg.copy()
# alpha = 0.6
# result[mask[:,:,3] != 0] = fg[mask[:,:,3] != 0] * alpha \
#     + eff[mask[:,:,3] != 0] * (1 - alpha)

# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Create Gif
import imageio
fg = cv2.imread('gifB.png')
mask = cv2.imread('gifA.png', cv2.IMREAD_UNCHANGED)

url = "https://media.giphy.com/media/Q33tjDjDyxMrSJiGjO/giphy.gif"
frames = imageio.mimread(imageio.core.urlopen(url).read(), '.gif')


fg_h, fg_w, fg_c = fg.shape
bg_h, bg_w, bg_c = frames[0].shape
top = int((bg_h-fg_h)/2)
left = int((bg_w-fg_w)/2)
bgs = [frame[top: top + fg_h, left:left + fg_w, 0:3] for frame in frames]

results = []
alpha = 0.3
for i in range(len(bgs)):
	result = fg.copy()
	result[mask[:,:,3] != 0] = alpha * result[mask[:,:,3] != 0]
	bgs[i][mask[:,:,3] == 0] = 0
	bgs[i][mask[:,:,3] != 0] = (1-alpha)*bgs[i][mask[:,:,3] != 0]
	result = result + bgs[i]
	results.append(result)

imageio.mimsave('result.gif', results)