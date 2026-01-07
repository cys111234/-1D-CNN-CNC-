import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')


class SignalAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = None
        self.current_channel = 0
        self.sampling_rate = 1000  # 默认采样率
        self.model = None
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('设备状态监测与预测系统')
        self.setGeometry(100, 50, 1600, 900)

        # 设置窗口图标
        self.setWindowIcon(QIcon('icon.png'))  # 需要准备图标文件

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout()

        # 左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)

        # 右侧显示区域
        display_area = self.create_display_area()
        main_layout.addWidget(display_area, 3)

        central_widget.setLayout(main_layout)

        # 初始化模型（模拟）
        self.init_model()

    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout()

        # 文件操作组
        file_group = QGroupBox("数据加载")
        file_layout = QVBoxLayout()

        self.btn_open = QPushButton("📁 打开文件")
        self.btn_open.setIcon(QIcon.fromTheme("document-open"))
        self.btn_open.clicked.connect(self.open_file)
        self.btn_open.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        file_layout.addWidget(self.btn_open)

        self.file_label = QLabel("未加载文件")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setStyleSheet("padding: 10px; border: 1px solid #ddd; border-radius: 4px;")
        file_layout.addWidget(self.file_label)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # 通道选择组
        channel_group = QGroupBox("通道选择")
        channel_layout = QVBoxLayout()

        self.channel_combo = QComboBox()
        self.channel_combo.currentIndexChanged.connect(self.channel_changed)
        channel_layout.addWidget(QLabel("选择传感器通道:"))
        channel_layout.addWidget(self.channel_combo)

        self.channel_info = QLabel("共 0 个通道")
        self.channel_info.setAlignment(Qt.AlignCenter)
        channel_layout.addWidget(self.channel_info)

        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)

        # 特征提取组
        feature_group = QGroupBox("特征提取")
        feature_layout = QVBoxLayout()

        self.btn_extract = QPushButton("🔍 提取特征")
        self.btn_extract.clicked.connect(self.extract_features)
        self.btn_extract.setEnabled(False)
        self.btn_extract.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        feature_layout.addWidget(self.btn_extract)

        feature_group.setLayout(feature_layout)
        layout.addWidget(feature_group)

        # 状态预测组
        predict_group = QGroupBox("状态预测")
        predict_layout = QVBoxLayout()

        self.btn_predict = QPushButton("🚀 开始预测")
        self.btn_predict.clicked.connect(self.predict_status)
        self.btn_predict.setEnabled(False)
        self.btn_predict.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        predict_layout.addWidget(self.btn_predict)

        # 状态指示灯
        self.status_light = QLabel()
        self.status_light.setFixedSize(100, 100)
        self.status_light.setAlignment(Qt.AlignCenter)
        self.set_status_light("unknown")
        predict_layout.addWidget(self.status_light, 0, Qt.AlignCenter)

        # 预测结果显示
        self.prediction_label = QLabel("等待预测...")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
                background-color: #f5f5f5;
            }
        """)
        predict_layout.addWidget(self.prediction_label)

        # 置信度显示
        self.confidence_label = QLabel("置信度: --%")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        predict_layout.addWidget(self.confidence_label)

        predict_group.setLayout(predict_layout)
        layout.addWidget(predict_group)

        # 系统信息
        info_group = QGroupBox("系统信息")
        info_layout = QVBoxLayout()

        self.sampling_rate_label = QLabel("采样率: 1000 Hz")
        self.data_points_label = QLabel("数据点数: 0")
        self.selected_channel_label = QLabel("当前通道: 无")

        for label in [self.sampling_rate_label, self.data_points_label, self.selected_channel_label]:
            label.setStyleSheet("padding: 5px;")
            info_layout.addWidget(label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_display_area(self):
        """创建显示区域"""
        widget = QWidget()
        layout = QVBoxLayout()

        # 标签页控件
        self.tab_widget = QTabWidget()

        # 时域图和频谱图标签页
        plots_tab = QWidget()
        plots_layout = QVBoxLayout()

        # 时域图
        self.time_figure = Figure(figsize=(10, 4))
        self.time_canvas = FigureCanvas(self.time_figure)
        self.time_ax = self.time_figure.add_subplot(111)
        plots_layout.addWidget(QLabel("📈 原始时域波形图"))
        plots_layout.addWidget(self.time_canvas)

        # 频谱图
        self.freq_figure = Figure(figsize=(10, 4))
        self.freq_canvas = FigureCanvas(self.freq_figure)
        self.freq_ax = self.freq_figure.add_subplot(111)
        plots_layout.addWidget(QLabel("📊 FFT频谱图"))
        plots_layout.addWidget(self.freq_canvas)

        plots_tab.setLayout(plots_layout)
        self.tab_widget.addTab(plots_tab, "信号可视化")

        # 特征表格标签页
        features_tab = QWidget()
        features_layout = QVBoxLayout()

        # 时域特征表格
        features_layout.addWidget(QLabel("⏱️ 时域特征"))
        self.time_features_table = QTableWidget()
        self.time_features_table.setColumnCount(2)
        self.time_features_table.setHorizontalHeaderLabels(["特征", "数值"])
        self.time_features_table.horizontalHeader().setStretchLastSection(True)
        features_layout.addWidget(self.time_features_table)

        # 频域特征表格
        features_layout.addWidget(QLabel("📡 频域特征"))
        self.freq_features_table = QTableWidget()
        self.freq_features_table.setColumnCount(2)
        self.freq_features_table.setHorizontalHeaderLabels(["特征", "数值"])
        self.freq_features_table.horizontalHeader().setStretchLastSection(True)
        features_layout.addWidget(self.freq_features_table)

        features_tab.setLayout(features_layout)
        self.tab_widget.addTab(features_tab, "特征提取")

        # 数据预览标签页
        preview_tab = QWidget()
        preview_layout = QVBoxLayout()

        preview_layout.addWidget(QLabel("📋 数据预览"))
        self.data_preview_table = QTableWidget()
        preview_layout.addWidget(self.data_preview_table)

        preview_tab.setLayout(preview_layout)
        self.tab_widget.addTab(preview_tab, "数据预览")

        layout.addWidget(self.tab_widget)
        widget.setLayout(layout)
        return widget

    def set_status_light(self, status):
        """设置状态指示灯"""
        colors = {
            "unknown": "#cccccc",
            "initial": "#FFEB3B",  # 黄色-初期磨损
            "normal": "#4CAF50",  # 绿色-正常磨损
            "severe": "#F44336"  # 红色-严重磨损
        }

        # 创建圆形状态灯
        pixmap = QPixmap(100, 100)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(colors[status]))
        painter.setPen(QPen(Qt.black, 2))
        painter.drawEllipse(10, 10, 80, 80)
        painter.end()

        self.status_light.setPixmap(pixmap)

    def open_file(self):
        """打开数据文件"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "打开数据文件", "",
            "CSV文件 (*.csv);;文本文件 (*.txt);;所有文件 (*.*)"
        )

        if file_name:
            try:
                # 读取数据
                self.data = pd.read_csv(file_name)

                # 更新文件信息
                self.file_label.setText(f"已加载: {file_name.split('/')[-1]}")

                # 更新通道选择
                self.channel_combo.clear()
                for i, col in enumerate(self.data.columns):
                    self.channel_combo.addItem(f"通道 {i}: {col}")

                # 更新系统信息
                self.channel_info.setText(f"共 {len(self.data.columns)} 个通道")
                self.data_points_label.setText(f"数据点数: {len(self.data)}")

                # 启用按钮
                self.btn_extract.setEnabled(True)
                self.btn_predict.setEnabled(True)

                # 更新数据预览
                self.update_data_preview()

                # 自动选择第一个通道并绘图
                if len(self.data.columns) > 0:
                    self.current_channel = 0
                    self.plot_signals()

                QMessageBox.information(self, "成功", f"成功加载文件！\n数据维度: {self.data.shape}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载文件失败:\n{str(e)}")

    def channel_changed(self, index):
        """通道改变事件"""
        if self.data is not None and index >= 0:
            self.current_channel = index
            self.selected_channel_label.setText(f"当前通道: {index} ({self.data.columns[index]})")
            self.plot_signals()

    def plot_signals(self):
        """绘制时域图和频谱图"""
        if self.data is None or self.current_channel >= len(self.data.columns):
            return

        try:
            # 获取当前通道数据
            signal_data = self.data.iloc[:, self.current_channel].values

            # 绘制时域图
            self.time_ax.clear()
            time = np.arange(len(signal_data)) / self.sampling_rate
            self.time_ax.plot(time, signal_data, 'b-', linewidth=1)
            self.time_ax.set_xlabel('时间 (s)')
            self.time_ax.set_ylabel('幅值')
            self.time_ax.set_title(f'时域波形 - 通道 {self.current_channel}')
            self.time_ax.grid(True, alpha=0.3)
            self.time_canvas.draw()

            # 计算并绘制频谱图
            self.freq_ax.clear()
            n = len(signal_data)
            fft_result = np.fft.fft(signal_data)
            freqs = np.fft.fftfreq(n, 1 / self.sampling_rate)

            # 取正频率部分
            positive_mask = freqs >= 0
            positive_freqs = freqs[positive_mask]
            positive_fft = np.abs(fft_result[positive_mask]) / n

            self.freq_ax.plot(positive_freqs, positive_fft, 'r-', linewidth=1)
            self.freq_ax.set_xlabel('频率 (Hz)')
            self.freq_ax.set_ylabel('幅值')
            self.freq_ax.set_title(f'频谱图 - 通道 {self.current_channel}')
            self.freq_ax.grid(True, alpha=0.3)
            self.freq_ax.set_xlim([0, self.sampling_rate / 2])
            self.freq_canvas.draw()

        except Exception as e:
            print(f"绘图错误: {str(e)}")

    def extract_features(self):
        """提取特征"""
        if self.data is None:
            return

        try:
            # 获取当前通道数据
            signal_data = self.data.iloc[:, self.current_channel].values

            # 计算时域特征
            time_features = self.calculate_time_domain_features(signal_data)

            # 更新时域特征表格
            self.update_features_table(self.time_features_table, time_features)

            # 计算频域特征
            freq_features = self.calculate_frequency_domain_features(signal_data)

            # 更新频域特征表格
            self.update_features_table(self.freq_features_table, freq_features)

        except Exception as e:
            QMessageBox.warning(self, "警告", f"特征提取失败:\n{str(e)}")

    def calculate_time_domain_features(self, signal):
        """计算时域特征"""
        features = {}

        # 均值
        features["均值"] = np.mean(signal)
        # 方差
        features["方差"] = np.var(signal)
        # 均方根
        features["均方根"] = np.sqrt(np.mean(signal ** 2))
        # 峭度
        features["峭度"] = np.mean((signal - np.mean(signal)) ** 4) / (np.std(signal) ** 4)
        # 峰峰值
        features["峰峰值"] = np.max(signal) - np.min(signal)
        # 峰值
        features["峰值"] = np.max(np.abs(signal))
        # 波形因子
        features["波形因子"] = np.sqrt(np.mean(signal ** 2)) / np.mean(np.abs(signal)) if np.mean(
            np.abs(signal)) != 0 else 0
        # 脉冲因子
        features["脉冲因子"] = np.max(np.abs(signal)) / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) != 0 else 0

        return features

    def calculate_frequency_domain_features(self, signal):
        """计算频域特征"""
        features = {}

        n = len(signal)
        # 计算FFT
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1 / self.sampling_rate)

        # 取正频率部分
        positive_mask = freqs >= 0
        positive_freqs = freqs[positive_mask]
        positive_fft = np.abs(fft_result[positive_mask])

        if len(positive_freqs) > 0:
            # 主频
            main_freq_idx = np.argmax(positive_fft)
            features["主频 (Hz)"] = positive_freqs[main_freq_idx]

            # 频率重心
            if np.sum(positive_fft) != 0:
                features["频率重心 (Hz)"] = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
            else:
                features["频率重心 (Hz)"] = 0

            # 均方根频率
            if np.sum(positive_fft) != 0:
                features["均方根频率 (Hz)"] = np.sqrt(
                    np.sum((positive_freqs ** 2) * positive_fft) / np.sum(positive_fft))
            else:
                features["均方根频率 (Hz)"] = 0

            # 频率方差
            if np.sum(positive_fft) != 0:
                freq_center = features["频率重心 (Hz)"]
                features["频率方差"] = np.sum(((positive_freqs - freq_center) ** 2) * positive_fft) / np.sum(
                    positive_fft)
            else:
                features["频率方差"] = 0

        return features

    def update_features_table(self, table, features):
        """更新特征表格"""
        table.setRowCount(len(features))
        for i, (key, value) in enumerate(features.items()):
            table.setItem(i, 0, QTableWidgetItem(key))
            if isinstance(value, float):
                table.setItem(i, 1, QTableWidgetItem(f"{value:.6f}"))
            else:
                table.setItem(i, 1, QTableWidgetItem(str(value)))

    def update_data_preview(self):
        """更新数据预览"""
        if self.data is not None:
            self.data_preview_table.setRowCount(min(50, len(self.data)))
            self.data_preview_table.setColumnCount(min(10, len(self.data.columns)))

            # 设置列标题
            self.data_preview_table.setHorizontalHeaderLabels(
                [f"通道 {i}" for i in range(min(10, len(self.data.columns)))]
            )

            # 填充数据
            for i in range(min(50, len(self.data))):
                for j in range(min(10, len(self.data.columns))):
                    self.data_preview_table.setItem(
                        i, j,
                        QTableWidgetItem(f"{self.data.iloc[i, j]:.6f}")
                    )

    def init_model(self):
        """初始化模型（模拟）"""
        # 这里应该加载预训练的1D-CNN模型
        # 为了演示，我们创建一个模拟的模型
        print("模型初始化完成（模拟）")

    def predict_status(self):
        """预测状态"""
        if self.data is None:
            return

        try:
            # 这里应该是实际的模型预测代码
            # 为了演示，我们随机生成一个预测结果

            # 模拟预测过程
            import time
            self.prediction_label.setText("预测中...")
            self.set_status_light("unknown")
            QApplication.processEvents()
            time.sleep(1)  # 模拟计算时间

            # 模拟预测结果
            import random
            status_options = [
                ("初期磨损 (Initial Wear)", "initial", 0.85),
                ("正常磨损 (Normal Wear)", "normal", 0.92),
                ("严重磨损 (Severe Wear)", "severe", 0.78)
            ]

            status_text, status_code, confidence = random.choice(status_options)

            # 更新预测结果
            self.prediction_label.setText(status_text)
            self.confidence_label.setText(f"置信度: {confidence * 100:.1f}%")
            self.set_status_light(status_code)

            # 根据状态设置不同的样式
            if status_code == "initial":
                self.prediction_label.setStyleSheet("""
                    QLabel {
                        font-size: 16px;
                        font-weight: bold;
                        padding: 15px;
                        border-radius: 8px;
                        background-color: #FFF9C4;
                        color: #F57C00;
                        border: 2px solid #FFB300;
                    }
                """)
            elif status_code == "normal":
                self.prediction_label.setStyleSheet("""
                    QLabel {
                        font-size: 16px;
                        font-weight: bold;
                        padding: 15px;
                        border-radius: 8px;
                        background-color: #C8E6C9;
                        color: #388E3C;
                        border: 2px solid #4CAF50;
                    }
                """)
            else:  # severe
                self.prediction_label.setStyleSheet("""
                    QLabel {
                        font-size: 16px;
                        font-weight: bold;
                        padding: 15px;
                        border-radius: 8px;
                        background-color: #FFCDD2;
                        color: #D32F2F;
                        border: 2px solid #F44336;
                    }
                """)

            # 显示详细分析报告
            report = f"""
            预测完成！

            设备状态: {status_text}
            置信度: {confidence * 100:.1f}%

            建议措施:
            {self.get_recommendation(status_code)}
            """

            QMessageBox.information(self, "预测结果", report)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测失败:\n{str(e)}")

    def get_recommendation(self, status_code):
        """获取建议措施"""
        recommendations = {
            "initial": "• 设备处于初期磨损阶段\n• 建议加强监测频率\n• 检查润滑系统",
            "normal": "• 设备运行正常\n• 按计划进行维护\n• 继续常规监测",
            "severe": "• 设备磨损严重\n• 建议立即停机检修\n• 更换磨损部件\n• 分析故障原因"
        }
        return recommendations.get(status_code, "无建议")


def main():
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    # 设置字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # 创建并显示主窗口
    window = SignalAnalysisApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()