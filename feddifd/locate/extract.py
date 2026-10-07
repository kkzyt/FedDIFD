import cv2
import numpy as np

# 读取两张图片（确保两张图片分辨率相同）
bbox_img = cv2.imread('example-my/bbox.jpg')
target_img = cv2.imread('example-my/ILSVRC2012_val_00009379.JPEG')

print(bbox_img)

# 确认尺寸一致
assert bbox_img.shape == target_img.shape, "两张图片尺寸不一致！"

# 转灰度得到掩膜
bbox_gray = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(bbox_gray, 240, 255, cv2.THRESH_BINARY)

# 找到掩膜中白色区域的轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    print("没有检测到白色区域！")
    exit()

# 找到最大轮廓对应的边界框（最小矩形）
x, y, w, h = cv2.boundingRect(contours[0])

# 从目标图片中裁剪对应区域
cropped_img = target_img[y:y+h, x:x+w]

# 同时裁剪掩膜区域以确认正确（可选）
cropped_mask = mask[y:y+h, x:x+w]

# 把掩膜用作透明度或者把非掩膜区域变为白色或其他颜色
# 这里示例是用掩膜把非掩膜区域变为白色
cropped_img_mask_bool = cropped_mask.astype(bool)
final_img = np.full_like(cropped_img, 255)  # 创建白色背景
final_img[cropped_img_mask_bool] = cropped_img[cropped_img_mask_bool]

# 保存结果
cv2.imwrite('example-my/extracted_result.JPEG', final_img)

print("提取完成，结果已保存为 extracted_result.JPEG")
