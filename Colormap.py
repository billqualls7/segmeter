
class Colormap:
    # 定义类属性，每个属性对应一个BGR颜色值
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    waterblue = (251, 255, 108)
    yellow = (0, 255, 255)
    purple = (128, 0, 128)
    cyan = (255, 255, 0)
    magenta = (255, 0, 255)
    orange = (0, 165, 255)
    black = (0, 0, 0)
    white = (255, 255, 255)
    brown = (165, 42, 42)
    light_blue = (173, 216, 230)
    gray = (128, 128, 128)
    

    def __init__(self):
        # 这里可以添加初始化代码，如果需要的话
        self.color_map = [
            self.black, self.red, self.green, self.blue, self.waterblue, self.yellow,
            self.purple, self.cyan, self.magenta, self.orange,
            self.white, self.brown, self.light_blue, self.gray
        ]


    def get_color(self, class_id):
        # 返回类的颜色，如果超出范围，则返回黑色
        if class_id < len(self.color_map):
            return self.color_map[class_id]
        else:
            return self.black  # 如果类别超出范围，返回黑色