# python3.7;java8:jdk1.8.0_351;neo4j3.5.34(add a plugin:neosemantics-3.5.0.4_2.jar)
# neo4j载入owl CALL semantics.importRDF('file:///C:/Users/50563/Desktop/pipe.turtle', 'RDF/XML',{handleVocabUris: "IGNORE"})
import base64
import io
from flask import Flask, jsonify, render_template, request, redirect, url_for, make_response
from keyFrame.key_frame_detector import keyframeDetection
import glob
import json
from model_detect import *
from jpype_drools import *
from flask_cors import CORS, cross_origin
from ultralytics import YOLO
# 开启jvm,用于运行drools
startjvm()
# 连接数据库
# graph = Graph('http://localhost:7474/', auth=('neo4j', 'root'))

app = Flask(__name__)
cors = CORS(app)

# global变量
IMAGE_FLAG = False
VIDEO_FLAG = False
advice_list = []
upload_files_path = []
identify_images = []
keyFramesIndices = []
neo4j_data = None
y57_c = {}
kf_c = {}
fr_c = {}
yx_c = {}
y8_c = {}
index_version = "index_cn.html"  # 默认语言

def run_yolov_8(source, weights, imgsz, conf_thres, iou_thres, device, line_thickness, hide_labels, hide_conf):
    # Add your YOLOv8 inference logic here
    # This might involve loading the model, running inference, and processing the output
    print(f"Running YOLOv8 on {source} with weights: {weights}, image size: {imgsz}")
    # Example logic here...

@app.route('/disease-detection/diseaseAnalysis', methods=['POST', 'OPTIONS'])
@cross_origin()
def diseaseAnalysis():
    print("diseaseAnalysis...")
    # 获取前端参数和图片
    modelType = request.form.get("modelType")
    deviceType = request.form.get("deviceType")
    files = request.files.getlist('files')  # 获取批量图片
    upload_files_path_s = []
    identify_images_s = []
    image_bytes_list = []

    # 保存图片
    upload_path = './static/uploads'
    for f in files:
        filename_without_extension = os.path.splitext(f.filename)[0]
        f.save(upload_path + "/" + filename_without_extension + ".jpg")
        upload_files_path_s.append(upload_path + "/" + filename_without_extension + ".jpg")
    print("upload_files_path_s", upload_files_path_s)
    if modelType == "YoloX":
        for upload_file_path in upload_files_path_s:
            print("upload_file_path", upload_files_path)
            run_yolox(path=upload_file_path)
            with open('./inference/output/identify_image.txt', "r+", encoding='UTF-8') as f:
                identify_images_s.append(f.readline())
                f.truncate(0)
            f.close()

    print(identify_images_s)

    for image_path in identify_images_s:
        # 读取 JPG 图片到内存中
        with open(image_path, 'rb') as f:
            image_data = io.BytesIO(f.read())

        # 加载图片数据并转换为 base64 编码字符串
        with Image.open(image_data) as img:
            buffered = io.BytesIO()
            img.save(buffered, format='JPEG')
            encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 创建响应对象并添加图片数据
    response_body = {
        'status': 'success',
        'message': 'post from flask',
        'data': {
            'image': encoded
        }

    }
    response_headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Credentials': 'true'
    }

    # 返回 HTTP 响应
    response = make_response((response_body, 200, response_headers))
    return response


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template(index_version, upload_files_path=upload_files_path, identify_images=identify_images,
                           advice_list=advice_list, VIDEO_FLAG=VIDEO_FLAG, neo4j_data=neo4j_data)


# 中英文切换
@app.route('/index_cn')
def index_cn():
    global index_version
    index_version = "index_cn.html"
    return render_template(index_version, upload_files_path=upload_files_path, identify_images=identify_images,
                           advice_list=advice_list, VIDEO_FLAG=VIDEO_FLAG, neo4j_data=neo4j_data)


@app.route(('/index_en'))
def index_en():
    global index_version
    index_version = "index_en.html"
    return render_template(index_version, upload_files_path=upload_files_path, identify_images=identify_images,
                           advice_list=advice_list, VIDEO_FLAG=VIDEO_FLAG, neo4j_data=neo4j_data)


@app.route('/refresh', methods=['POST', 'GET'])
def refresh():
    """Video streaming home page."""
    global upload_files_path, identify_images, advice_list
    upload_files_path = []
    identify_images = []
    advice_list = []
    with open('./inference/output/identify_image.txt', "r+", encoding='UTF-8') as f:
        f.truncate(0)
    f.close()
    for root, dirs, files in os.walk("./static/output", topdown=False):
        # 第一步：删除文件
        for name in files:
            os.remove(os.path.join(root, name))  # 删除文件
        # 第二步：删除空文件夹
        for name in dirs:
            # 预留exp文件夹为空，方便后续处理drools.txt
            if name == "exp" or name == "faster-rcnn" or name == "yolox" or name == "yolov5":
                continue
            os.rmdir(os.path.join(root, name))  # 删除一个空目录
    for root, dirs, files in os.walk("./static/keyFrames", topdown=False):
        # 第一步：删除文件
        for name in files:
            os.remove(os.path.join(root, name))  # 删除文件
    # return render_template(index_version, upload_files_path=upload_files_path,identify_images=identify_images,advice_list=advice_list,VIDEO_FLAG=VIDEO_FLAG)
    return redirect(url_for('index'))
#YOLOV 8 COnfiguration
@app.route('/yolov8_configuration', methods=['POST'])
def yolov8_configuration():
    global y8_c
    result = request.form

    # Extracting relevant parameters from the form
    weights = result.get("yolov8_weights")
    imgsz = result.get("imgsz")
    imgsz = imgsz.replace("(", "").replace(")", "").strip().split(',')
    conf_thres = result.get("conf_thres")
    iou_thres = result.get("iou_thres")
    device = result.get("device")
    line_thickness = result.get("line_thickness")
    hide_labels = result.get("hide_labels")
    hide_conf = result.get("hide_conf")

    # Store configurations in the global variable
    y8_c["weights"] = weights
    y8_c["imgsz"] = imgsz
    y8_c["conf_thres"] = conf_thres
    y8_c["iou_thres"] = iou_thres
    y8_c["device"] = device
    y8_c["line_thickness"] = line_thickness
    y8_c["hide_labels"] = hide_labels
    y8_c["hide_conf"] = hide_conf
    
    return redirect(url_for('index'))


@app.route('/yolov57_configuration', methods=['POST'])
def yolov57_configuration():
    global y57_c
    result = request.form
    weights = result.get("yolov5_weights")

    imgsz = result.get("imgsz")
    imgsz = imgsz.replace("(", "")
    imgsz = imgsz.replace(")", "")
    imgsz = imgsz.strip().split(',')

    conf_thres = result.get("conf_thres")
    iou_thres = result.get("iou_thres")
    device = result.get("device")
    line_thickness = result.get("line_thickness")
    hide_labels = result.get("hide_labels")
    hide_conf = result.get("hide_conf")
    y57_c["weights"] = weights
    y57_c["imgsz"] = imgsz
    y57_c["conf_thres"] = conf_thres
    y57_c["iou_thres"] = iou_thres
    y57_c["device"] = device
    y57_c["line_thickness"] = line_thickness
    y57_c["hide_labels"] = hide_labels
    y57_c["hide_conf"] = hide_conf
    return redirect(url_for('index'))


@app.route("/yolox_configuration", methods=['POST'])
def yolox_configuration():
    global yx_c
    result = request.form
    yx_c = result.to_dict()
    print(yx_c)
    return redirect(url_for('index'))


@app.route("/faster_rcnn_configuration", methods=['POST'])
def faster_rcnn_configuration():
    global fr_c
    result = request.form
    fr_c = result.to_dict()
    print(fr_c)
    return redirect(url_for('index'))


@app.route("/keyframe_configuration", methods=['POST'])
def keyframe_configuration():
    global kf_c
    result = request.form
    kf_c = result.to_dict()
    print(kf_c)
    return redirect(url_for('index'))


@app.route('/upload', methods=['POST'])
def upload():
    # 全局变量
    global IMAGE_FLAG, upload_files_path, VIDEO_FLAG
    # 获取图片并保存
    f = request.files.get("file")
    print(f)
    if f.filename.endswith(
            (".avi", ".wmv", ".mpeg", ".mp4", ".m4v", ".mov", ".asf", ".flv", ".f4v", ".rmvb", ".rm", ".3gp", ".vob")):
        VIDEO_FLAG = True
        IMAGE_FLAG = False
    elif f.filename.endswith(('.bmp', '.jpg', '.png', '.tif', '.gif', '.pcx', '.tga', '.exif', '.fpx', '.svg', '.psd',
                              '.cdr', '.pcd', '.dxf', '.ufo', '.eps', '.ai', '.raw', '.WMF', '.webp', '.avif',
                              '.apng')):
        # IMAGE_FLAG用来表示传输的是文件
        IMAGE_FLAG = True
        VIDEO_FLAG = False
    # 当前项目所在路径
    upload_path = './static/uploads'
    # 如果不存在以上路径，则创建相应的文件夹
    if not os.path.exists(upload_path):
        os.mkdir(upload_path)
    # 注意：没有的文件夹一定要先创建，不然会提示没有该路径 #保存原始图像至./static/uploads
    # 上传的图片以jpg格式保存，如果png格式，否则可能会报cannot unpack non-iterable PngImageFile object
    filename_without_extension = os.path.splitext(f.filename)[0]
    f.save(upload_path + "/" + filename_without_extension + ".jpg")
    upload_files_path.append(upload_path + "/" + filename_without_extension + ".jpg")
    print("upload_files_path", upload_files_path)
    return "none"

@app.route("/detect", methods=['POST'])
def detect():
    global identify_images, keyFramesIndices
    result_model = request.form
    print("result_model", result_model)

    for upload_file_path in upload_files_path:
        if IMAGE_FLAG:
            # Handling YOLOv5, Faster R-CNN, YOLOX, and now YOLOv8
            if y57_c:
                print("配置yolov5后")
                print(y57_c)
                run_yolov_5_7(source=upload_file_path, weights=y57_c["weights"],
                              imgsz=(int(y57_c["imgsz"][0]), int(y57_c["imgsz"][1])),
                              conf_thres=float(y57_c["conf_thres"]),
                              iou_thres=float(y57_c["iou_thres"]), device=y57_c["device"],
                              line_thickness=int(y57_c["line_thickness"]), hide_labels=y57_c["hide_labels"],
                              hide_conf=y57_c["hide_conf"], yolo_version=5)
            elif y8_c:  # Check if YOLOv8 configurations are set
                print("配置YOLOv8参数")
                run_yolov_8(source=upload_file_path, weights=y8_c["weights"],
                             imgsz=(int(y8_c["imgsz"][0]), int(y8_c["imgsz"][1])),
                             conf_thres=float(y8_c["conf_thres"]),
                             iou_thres=float(y8_c["iou_thres"]), device=y8_c["device"],
                             line_thickness=int(y8_c["line_thickness"]), hide_labels=y8_c["hide_labels"],
                             hide_conf=y8_c["hide_conf"])
            elif fr_c:
                print("配置faster_rcnn")
                run_faster_rcnn(source=upload_file_path, nms_iou=float(fr_c["nms_iou"]),
                                confidence=float(fr_c["confidence"]), thickness=int(fr_c["thickness"]))
            elif yx_c:
                print("配置YOLOX参数")
                run_yolox(path=upload_file_path, nms=float(yx_c["nms"]), conf=float(yx_c["conf"]),
                          device=yx_c["device"], tsize=int(yx_c["tsize"]))
            else:
                print("采用默认参数进行缺陷检测")
                if result_model["model_type"] == "YOLOv5":
                    run_yolov_5_7(source=upload_file_path, yolo_version=5)
                elif result_model["model_type"] == "YOLOv8":  # Add this condition
                    run_yolov_8(source=upload_file_path)
                elif result_model["model_type"] == "YOLOX":
                    run_yolox(path=upload_file_path)
                elif result_model["model_type"] == "Faster R-CNN":
                    run_faster_rcnn(source=upload_file_path)
                elif result_model["model_type"] == "YOLOv7":
                    run_yolov_5_7(source=upload_file_path, yolo_version=7)
                else:
                    run_yolov_5_7(source=upload_file_path, yolo_version=7)
            # The rest of your code for handling identify_images...

        elif VIDEO_FLAG:
            # Your existing video processing code...
            pass

    print("identify images path:", identify_images)
    return redirect(url_for('index'))


@app.route('/get_repairAdvice', methods=['POST', 'GET'])
def get_repairAdvice():
    global advice_list, index_version
    i = request.args.get("i")
    excel_path = glob.glob("static/output/exp" + i + "/*.xls")
    print("excel_path", excel_path)

    # Example: Running YOLOv8 for defect detection
    source = f"static/images/{i}.jpg"  # Update the path to the corresponding image
    weights = "D:/yolov8/runs/detect/train126/weights/best.pt"  # Update the path to your weights file
    imgsz = 640  # Image size for YOLOv8
    conf_thres = 0.25  # Confidence threshold
    iou_thres = 0.45  # IoU threshold
    device = "cpu"  # Change to "cuda" if using GPU
    line_thickness = 2  # Thickness of bounding box lines
    hide_labels = False  # Show/hide labels
    hide_conf = False  # Show/hide confidence

    # Call the YOLOv8 function
    run_yolov_8(source, weights, imgsz, conf_thres, iou_thres, device, line_thickness, hide_labels, hide_conf)

    # 中英文版本切换
    if index_version == "index_cn.html":
        if excel_path:  # 正常通过提取公式 BX
            # 取出推理得到的参数
            jpype_run_drools(excel_path[0], "static/output/exp" + i + "/")
            while os.path.exists("./static/output/exp" + i + "/drools.txt") == False:
                pass
            if os.path.exists("./static/output/exp" + i + "/drools.txt") == False:
                return render_template("index_cn.html", upload_files_path=upload_files_path,
                                       identify_images=identify_images)
            while os.path.getsize("./static/output/exp" + i + "/drools.txt") == 0:
                pass
            file_object = open("./static/output/exp" + i + "/drools.txt", 'r+', encoding='UTF-8')
            try:
                advice_list = []
                temp_dict = {}
                temp_dict_neo4j = {}
                # temp_dict["attribute"] = "Order"
                temp_dict["attribute"] = "顺序"
                temp_dict["value"] = i
                advice_list.append(temp_dict)
                if VIDEO_FLAG:
                    temp_dict = {}
                    # temp_dict["attribute"] = "Number Of Frames"
                    temp_dict["attribute"] = "视频帧次序"
                    temp_dict["value"] = keyFramesIndices[int(i) - 1]
                    advice_list.append(temp_dict)
                for line in file_object:
                    temp_dict = {}
                    v = line.strip().split(':')
                    temp_dict_neo4j[v[0]] = v[1]
                    if v[0] == "E":
                        temp_dict["attribute"] = v[0] + "(管道重要性参数,默认值:3)"
                    elif v[0] == "K":
                        temp_dict["attribute"] = v[0] + "(区域重要性参数,默认值:6)"
                    elif v[0] == "T":
                        temp_dict["attribute"] = v[0] + "(土壤质量影响参数,默认值:8)"
                    elif v[0] == "smax":
                        temp_dict["attribute"] = v[0] + "(管道区域损坏形势参数, 最大值)"
                    elif v[0] == "s":
                        temp_dict["attribute"] = v[0] + "(管道区域损坏形势参数, 平均值)"
                    elif v[0] == "f":
                        temp_dict["attribute"] = v[0] + "(管道区域结构性缺陷参数)"
                    elif v[0] == "ri":
                        temp_dict["attribute"] = v[0] + "(分段修复指数)"
                    elif v[0] == "diameter":
                        temp_dict["attribute"] = v[0] + "(单位: mm)"
                    else:
                        temp_dict["attribute"] = v[0]
                    temp_dict["value"] = v[1]
                    advice_list.append(temp_dict)
            finally:
                file_object.close()
            # print("temp_dict_neo4j", temp_dict_neo4j)
            print("advice_list", advice_list)
        else:
            # 未检测到管道缺陷
            advice_list = []
            temp_dict = {}
            # temp_dict["attribute"] = "Order"
            temp_dict["attribute"] = "顺序"
            temp_dict["value"] = i
            advice_list.append(temp_dict)
            if VIDEO_FLAG:
                temp_dict = {}
                # temp_dict["attribute"] = "Number Of Frames"
                temp_dict["attribute"] = "视频帧次序"
                temp_dict["value"] = keyFramesIndices[int(i) - 1]
                advice_list.append(temp_dict)
            temp_dict = {}
            temp_dict["attribute"] = "提示"
            temp_dict["value"] = "图像中不存在缺陷或者该缺陷类型正在开发中"
            advice_list.append(temp_dict)

    elif index_version == "index_en.html":
        if excel_path:
            jpype_run_drools(excel_path[0], "static/output/exp" + i + "/")
            while os.path.exists("./static/output/exp" + i + "/drools.txt") == False:
                pass
            if os.path.exists("./static/output/exp" + i + "/drools.txt") == False:
                return render_template("index_cn.html", upload_files_path=upload_files_path,
                                       identify_images=identify_images)
            while os.path.getsize("./static/output/exp" + i + "/drools.txt") == 0:
                pass
            file_object = open("./static/output/exp" + i + "/drools.txt", 'r+', encoding='UTF-8')
            try:
                advice_list = []
                temp_dict = {}
                temp_dict_neo4j = {}
                temp_dict["attribute"] = "Order"
                temp_dict["value"] = i
                advice_list.append(temp_dict)
                if VIDEO_FLAG:
                    temp_dict = {}
                    temp_dict["attribute"] = "Number Of Frames"
                    temp_dict["value"] = keyFramesIndices[int(i) - 1]
                    advice_list.append(temp_dict)
                for line in file_object:
                    temp_dict = {}
                    v = line.strip().split(':')
                    temp_dict_neo4j[v[0]] = v[1]
                    if v[0] == "E":
                        temp_dict["attribute"] = v[0] + "(Pipeline importance parameters,default:3)"
                    elif v[0] == "K":
                        temp_dict["attribute"] = v[0] + "(Regional importance parameter,default:6)"
                    elif v[0] == "T":
                        temp_dict["attribute"] = v[0] + "(Soil quality influence parameter,default:8)"
                    elif v[0] == "smax":
                        temp_dict["attribute"] = v[0] + "(Pipeline section damage condition parameter, the most severe punishment value)"
                    elif v[0] == "s":
                        temp_dict["attribute"] = v[0] + "(Pipe section damage condition parameter, average score)"
                    elif v[0] == "f":
                        temp_dict["attribute"] = v[0] + "(Structural defect parameters of pipe section)"
                    elif v[0] == "ri":
                        temp_dict["attribute"] = v[0] + "(Segment repair index)"
                    elif v[0] == "diameter":
                        temp_dict["attribute"] = v[0] + "(Unit: mm)"
                    else:
                        temp_dict["attribute"] = v[0]
                    temp_dict["value"] = v[1]
                    advice_list.append(temp_dict)
            finally:
                file_object.close()
            print("advice_list", advice_list)
        else:
            # 未检测到管道缺陷
            advice_list = []
            temp_dict = {}
            temp_dict["attribute"] = "Order"
            temp_dict["value"] = i
            advice_list.append(temp_dict)
            if VIDEO_FLAG:
                temp_dict = {}
                temp_dict["attribute"] = "Number Of Frames"
                temp_dict["value"] = keyFramesIndices[int(i) - 1]
                advice_list.append(temp_dict)
            temp_dict = {}
            temp_dict["attribute"] = "Tips"
            temp_dict["value"] = "No defect is detected in the image or the defect category is still under development"
            advice_list.append(temp_dict)
    # return redirect(url_for('index'))
    # return (index_version, 204)
    return render_template(index_version, upload_files_path=upload_files_path, identify_images=identify_images,
                           advice_list=advice_list, VIDEO_FLAG=VIDEO_FLAG, neo4j_data=neo4j_data)

@app.route('/repair_advice_table')
def repair_advice_table():
    return render_template("repair_advice_table.html", advice_list=advice_list)
    '''
    # return redirect(url_for('index'))
    # return (index_version, 204)
    return render_template(index_version, upload_files_path=upload_files_path, identify_images=identify_images,
                           advice_list=advice_list, VIDEO_FLAG=VIDEO_FLAG, neo4j_data=neo4j_data)

@app.route('/repair_advice_table')
def repair_advice_table():
    return render_template("repair_advice_table.html", advice_list=advice_list)


'''
@app.route("/createnode")
def createnode():
    advice_list[""]
    temp = Node("Pipe", uri="image114514")
    graph.create(temp)
    return "create node success"

# MATCH (n:Pipe{uri: 'image1'}) DETACH DELETE (n) 直接在neo4j中删除
@app.route("/deletenode")
def deletenode():
    temp = Node("Pipe", uri="image")
    graph.delete(temp)
    return "delete node success"

# 查询neo4j数据库中的节点和关系，并保存成json文件，用于实现本体模型的可视化，只需执行一次此函数
@app.route("/search_all")
def search_all():
    # 定义data数组，存放节点信息
    data = []
    # 定义关系数组，存放节点间的关系
    links = []
    # 查询所有节点，并将节点信息取出存放在data数组中
    for n in graph.nodes:
        # 将节点信息转化为json格式，否则中文会不显示
        nodesStr = json.dumps(graph.nodes[n], ensure_ascii=False)
        print(nodesStr)
        # 取出节点的name
        node_name = json.loads(nodesStr)['uri']
        if "http:" in node_name or "genid-" in node_name:
            continue
        # 构造字典，存储单个节点信息
        dict = {
            'name': node_name,
            'symbolSize': 50,
            'category': 'Object'
        }
        # 将单个节点信息存放在data数组中
        data.append(dict)
    # 查询所有关系，并将所有的关系信息存放在links数组中
    rps = graph.relationships
    for r in rps:
        print(rps[r])
        # 取出开始节点的name
        source = str(rps[r].start_node['uri'])
        # 取出结束节点的name
        target = str(rps[r].end_node['uri'])
        # 取出开始节点的结束节点之间的关系
        name = str(type(rps[r]).__name__)
        # 构造字典存储单个关系信息
        dict = {
            'source': source,
            'target': target,
            'name': name
        }
        # 将单个关系信息存放进links数组中
        links.append(dict)
    # # 输出所有节点信息
    # for item in data:
    #     print(item)
    # # 输出所有关系信息
    # for item in links:
    #     print(item)
    # 将所有的节点信息和关系信息存放在一个字典中
    neo4j_data = {
        'data': data,
        'links': links
    }
    with open("static/owl_json/owl_nodes_relationships.json", "w", encoding='utf-8') as f:
        # 将字典转化json格式
        neo4j_data = json.dump(neo4j_data,f)
    return render_template("index_cn.html")
if __name__ == '__main__':
    app.run(host='127.0.0.1', threaded=True, port=5000)
