import os
import sys
import cv2
import xml.etree.ElementTree as ET
from ultralytics import YOLO


def create_xml(predictions, image_shape, image_name, output_path):
    # 创建 XML 的根节点
    annotation = ET.Element("annotation")

    # 添加 source 节点
    source = ET.SubElement(annotation, "source")
    filename = ET.SubElement(source, "filename")
    filename.text = image_name
    origin = ET.SubElement(source, "origin")
    origin.text = "Unknown"  # 可以根据实际情况修改

    # 添加 size 节点
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image_shape[1])
    height = ET.SubElement(size, "height")
    height.text = str(image_shape[0])
    depth = ET.SubElement(size, "depth")
    depth.text = str(image_shape[2])

    # 添加 objects 节点
    objects = ET.SubElement(annotation, "objects")

    for pred in predictions:
        obj = ET.SubElement(objects, "object")

        coordinate = ET.SubElement(obj, "coordinate")
        coordinate.text = "pixel"

        type_ = ET.SubElement(obj, "type")
        type_.text = "rectangle"

        description = ET.SubElement(obj, "description")
        description.text = "None"

        possibleresult = ET.SubElement(obj, "possibleresult")
        name = ET.SubElement(possibleresult, "name")
        name.text = pred['name']

        points = ET.SubElement(obj, "points")
        for point in pred['points']:
            point_tag = ET.SubElement(points, "point")
            point_tag.text = f"{point[0]},{point[1]}"

    # 将 XML 文件保存到输出路径
    tree = ET.ElementTree(annotation)
    xml_filename = os.path.join(output_path, os.path.splitext(image_name)[0] + '.xml')
    tree.write(xml_filename, encoding='utf-8', xml_declaration=True)


def main(input_path, output_path):
    # 加载 YOLOv8 模型
    model = YOLO("/workspace/best.pt")  # 替换为实际的权重文件路径

    # 确保输出路径存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 读取输入路径中的所有图像
    for image_name in os.listdir(input_path):
        image_path = os.path.join(input_path, image_name)
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
            continue

        # 使用 OpenCV 读取图像
        image = cv2.imread(image_path)

        # 使用模型进行预测
        results = model.predict(image)

        # 将预测结果转换为 XML 格式并保存
        predictions = []
        for result in results:
            for box in result.boxes:
                # 将预测结果格式化为 XML 需要的格式
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                class_name = model.names[int(box.cls)]
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]

                predictions.append({
                    'name': class_name,
                    'points': points
                })

        create_xml(predictions, image.shape, image_name, output_path)


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(input_path, output_path)
