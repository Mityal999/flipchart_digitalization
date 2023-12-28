import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw
import os
import json

# Инициализация глобальных переменных
reader = None
processor = None
model = None

def initialize_ocr():
    global reader, processor, model
    if reader is None:
        reader = easyocr.Reader(['ru','en'])
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def replace_text_with_rectangles(image_path, corner_radius=0.6):
    # Замена областей текста белыми прямоугольниками
    result = reader.readtext(image_path)
    coords_list = [x[0] for x in result]

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for coords in coords_list:
        upper_left = (coords[0][0], coords[0][1])  # Верхний левый угол
        lower_right = (coords[2][0], coords[2][1])  # Нижний правый угол
        draw.rounded_rectangle([upper_left, lower_right], corner_radius * 100, fill="white")
    
    return image, coords_list

def recognize_and_generate_xml(image, coords_list, name, xml_output_folder, font_size=20):
    # Распознавание текста и генерация XML-файла для draw.io
    recognized_text = []
    for coords in coords_list:
        cropped_image = image.crop([coords[0][0], coords[0][1], coords[2][0], coords[2][1]])
        pixel_values = processor(cropped_image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        recognized_text.append({"coordinates": coords, "text": generated_text})

    # Создание XML для draw.io
    xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml_content += '<mxfile host="app.diagrams.net">\n'
    xml_content += '  <diagram>\n'
    xml_content += '    <mxGraphModel>\n'
    xml_content += '      <root>\n'
    xml_content += '        <mxCell id="0" />\n'
    xml_content += '        <mxCell id="1" parent="0" />\n'

    cell_id = 1
    for item in recognized_text:
        x1, y1, x2, y2 = item["coordinates"][0][0], item["coordinates"][0][1], item["coordinates"][2][0], item["coordinates"][2][1]
        text = item["text"].replace("\n", "&lt;br&gt;")
        width = x2 - x1
        height = y2 - y1

        xml_content += f'        <mxCell id="cell{cell_id}" value="{text}" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize={font_size};" vertex="1" parent="1">\n'
        xml_content += f'          <mxGeometry x="{x1}" y="{y1}" width="{width}" height="{height}" as="geometry" />\n'
        xml_content += '        </mxCell>\n'
        cell_id += 1

    xml_content += '      </root>\n'
    xml_content += '    </mxGraphModel>\n'
    xml_content += '  </diagram>\n'
    xml_content += '</mxfile>\n'
    
    with open(f"{xml_output_folder}/{name}_text.xml", "w", encoding="utf-8") as file:
        file.write(xml_content)
    print("Completed text XML file creation for draw.io")


def recognize(name, tmp_folder, xml_output_folder):
    input_path = f'{tmp_folder}/{name}.jpg'

    initialize_ocr()

    modified_image, coords_list = replace_text_with_rectangles(input_path)
    modified_image.save(f"{tmp_folder}/{name}_cut.jpg")

    image = Image.open(input_path).convert("RGB")
    recognize_and_generate_xml(image, coords_list, name, xml_output_folder)


    print("Completed handwritten text recognition in image")