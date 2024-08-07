from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication
from PyQt5.Qt import QPainter, QPoint, QPen
from PyQt5.QtCore import Qt
from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton,QComboBox, QLabel, QSpinBox
from predict import get_predict
import sys


def main():
    app = QApplication(sys.argv)
    mainWidget = MainWidget()  # 新建一个主界面
    mainWidget.show()  # 显示主界面
    exit(app.exec_())  # 进入消息循环


class PaintBoard(QWidget):
    def __init__(self, Parent=None):
        super().__init__(Parent)
        self.__InitData()  # 先初始化数据，再初始化界面
        self.__InitView()
        self.setWindowTitle("画笔")

    def __InitData(self):
        self.__size = QSize(224, 224)

        # 新建QPixmap作为画板，尺寸为__size
        self.__board = QPixmap(self.__size)
        self.__board.fill(Qt.white)  # 用白色填充画板

        self.__IsEmpty = True  # 默认为空画板
        self.EraserMode = False  # 默认为禁用橡皮擦模式

        self.__lastPos = QPoint(0, 0)  # 上一次鼠标位置
        self.__currentPos = QPoint(0, 0)  # 当前的鼠标位置

        self.__painter = QPainter()  # 新建绘图工具

        self.__thickness = 15  # 默认画笔粗细为10px
        self.__penColor = QColor("black")  # 设置默认画笔颜色为黑色
        self.__colorList = QColor.colorNames()  # 获取颜色列表

    def __InitView(self):
        # 设置界面的尺寸为__size
        self.setFixedSize(self.__size)

    def Clear(self):
        # 清空画板
        self.__board.fill(Qt.white)
        self.update()
        self.__IsEmpty = True

    def IsEmpty(self):
        # 返回画板是否为空
        return self.__IsEmpty

    def GetContentAsQImage(self):
        # 获取画板内容（返回QImage）
        image = self.__board.toImage()
        return image

    def paintEvent(self, paintEvent):
        # 绘图事件
        # 绘图时必须使用QPainter的实例，此处为__painter
        # 绘图在begin()函数与end()函数间进行
        # begin(param)的参数要指定绘图设备，即把图画在哪里
        # drawPixmap用于绘制QPixmap类型的对象
        self.__painter.begin(self)
        # 0,0为绘图的左上角起点的坐标，__board即要绘制的图
        self.__painter.drawPixmap(0, 0, self.__board)
        self.__painter.end()

    def mousePressEvent(self, mouseEvent):
        # 鼠标按下时，获取鼠标的当前位置保存为上一次位置
        self.__currentPos = mouseEvent.pos()
        self.__lastPos = self.__currentPos

    def mouseMoveEvent(self, mouseEvent):
        # 鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.__currentPos = mouseEvent.pos()
        self.__painter.begin(self.__board)

        if self.EraserMode == False:
            # 非橡皮擦模式
            self.__painter.setPen(QPen(self.__penColor, self.__thickness))  # 设置画笔颜色，粗细
        else:
            # 橡皮擦模式下画笔为纯白色，粗细为10
            self.__painter.setPen(QPen(Qt.white, 10))

        # 画线
        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos

        self.update()  # 更新显示

    def mouseReleaseEvent(self, mouseEvent):
        self.__IsEmpty = False  # 画板不再为空


class MainWidget(QWidget):
    def __init__(self, Parent=None):
        super().__init__(Parent)
        self.__InitData()  # 先初始化数据，再初始化界面
        self.__InitView()

    def __InitData(self):
        """
        初始化成员变量
        """
        self.__paintBoard = PaintBoard(self)
        # 获取颜色列表(字符串类型)
        self.__colorList = QColor.colorNames()

    def __InitView(self):
        """
        初始化界面
        """
        self.setFixedSize(480, 320)
        self.setWindowTitle("手写汉字识别器")

        # 新建一个水平布局作为本窗体的主布局
        main_layout = QHBoxLayout(self)
        # 设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10)

        # 在主界面左侧放置画板
        main_layout.addWidget(self.__paintBoard)

        # 新建垂直子布局用于放置按键
        sub_layout = QVBoxLayout()

        # 设置此子布局和内部控件的间距为10px
        sub_layout.setContentsMargins(10, 10, 10, 10)

        self.__btn_Clear = QPushButton("清空画板")
        self.__btn_Clear.setParent(self)  # 设置父对象为本界面

        # 将按键按下信号与画板清空函数相关联
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear)
        sub_layout.addWidget(self.__btn_Clear)

        self.__btn_Quit = QPushButton("退出")
        self.__btn_Quit.setParent(self)  # 设置父对象为本界面
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)

        self.__btn_Save = QPushButton("预测")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)

        self.__textbox = QLineEdit(self)
        # self.__textbox.move(20, 20)
        # self.__textbox.resize(10,10)
        self.__textbox.setReadOnly(True)
        sub_layout.addWidget(self.__textbox)
        main_layout.addLayout(sub_layout)  # 将子布局加入主布局

    def __fillColorList(self, comboBox):
        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__paintBoard.ChangePenColor(color_str)

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)

    def on_btn_Save_Clicked(self): # 按钮点击事件触发
        image = self.__paintBoard.GetContentAsQImage()
        image.save('test.png')  # 默认保存为test.png文件
        ans = get_predict() # 调用函数进行预测
        self.__textbox.setText("预测结果为:" + ans)

    def Quit(self):
        self.close()


if __name__ == '__main__':
    main()

