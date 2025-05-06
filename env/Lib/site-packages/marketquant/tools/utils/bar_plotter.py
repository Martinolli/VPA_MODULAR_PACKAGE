import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os


class BarPlotter:
    def __init__(self, dataframe: pd.DataFrame, x_col: str, y_col: str,
                 figure_size=(12, 8), dpi=100, font_size=12, bar_width=0.8,
                 grid=True, grid_color='grey', grid_line_style="--", grid_line_width=0.5,
                 axis_color='black', tick_color='black', axis_line_width=1.0,
                 x_rotation=45, legend=True, title_font_size=16, title_font_color="navy",
                 xlabel_font_size=14, ylabel_font_size=14, label_color="purple"):
        """
        Initialize the BarChartPlotter with the given DataFrame and column names along with customizations.
        :param dataframe: A pandas DataFrame containing the data.
        :param x_col: The column name for the x-axis (e.g., 'strikePrice').
        :param y_col: The column name for the y-axis (e.g., 'gammaExposure').
        :param figure_size: Tuple for figure size (width, height).
        :param dpi: DPI for the figure.
        :param font_size: Font size for x and y labels.
        :param bar_width: Width of the bars.
        :param grid: Enable or disable grid.
        :param grid_color: Color of the grid lines.
        :param grid_line_style: Style of the grid lines (e.g., '--', '-').
        :param grid_line_width: Thickness of the grid lines.
        :param axis_color: Color of the axes.
        :param tick_color: Color of the tick marks.
        :param axis_line_width: Width of the axis lines.
        :param x_rotation: Rotation of the x-axis labels.
        :param legend: Whether to show the legend.
        :param title_font_size: Font size of the chart title.
        :param title_font_color: Color of the chart title.
        :param xlabel_font_size: Font size for the x-axis label.
        :param ylabel_font_size: Font size for the y-axis label.
        :param label_color: Color for both x and y labels.
        """
        self.dataframe = dataframe
        self.x_col = x_col
        self.y_col = y_col
        self.figure_size = figure_size
        self.dpi = dpi
        self.font_size = font_size
        self.bar_width = bar_width
        self.grid = grid
        self.grid_color = grid_color
        self.grid_line_style = grid_line_style
        self.grid_line_width = grid_line_width
        self.axis_color = axis_color
        self.tick_color = tick_color
        self.axis_line_width = axis_line_width
        self.x_rotation = x_rotation
        self.legend = legend
        self.title_font_size = title_font_size
        self.title_font_color = title_font_color
        self.xlabel_font_size = xlabel_font_size
        self.ylabel_font_size = ylabel_font_size
        self.label_color = label_color

    def validate_data(self):
        """
        Validate if the DataFrame contains the required columns.
        :return: True if valid, False otherwise.
        """
        required_columns = {self.x_col, self.y_col}
        if required_columns.issubset(self.dataframe.columns):
            print("Data validation successful. The DataFrame contains the required columns.")
            return True
        else:
            missing_columns = required_columns - set(self.dataframe.columns)
            print(f"Data validation failed. Missing columns: {missing_columns}")
            return False

    def add_logo(self, ax):
        """
        Add the MarketQuant logo to the bottom left of the chart.
        :param ax: The axis to add the logo to.
        """
        logo_path = 'marketquant/MarketQuant_Logo.png'
        if os.path.exists(logo_path):
            img = plt.imread(logo_path)
            imagebox = OffsetImage(img, zoom=0.1, alpha=0.8)
            ab = AnnotationBbox(imagebox, (0, 0), frameon=False, xycoords='axes fraction',
                                boxcoords="offset points", pad=0.5, xybox=(50, 50))
            ax.add_artist(ab)
        else:
            print(f"Logo image not found at {logo_path}")

    def plot_barchart(self, positive_color='blue', negative_color='red', title="Gamma Exposure",
                      line_data=None, line_color='green', background_color='white'):
        """
        Plot a bar chart with customizable colors for positive and negative y-values, and optional line data.
        :param positive_color: Color for positive values (default: 'blue').
        :param negative_color: Color for negative values (default: 'red').
        :param title: Title of the bar chart.
        :param line_data: Optional dictionary with keys as x-values and values as line y-values.
        :param line_color: Color of the line (default: 'green').
        :param background_color: Background color of the chart.
        """
        if not self.validate_data():
            return

        # Ensure all data points are plotted
        all_data = self.dataframe

        # Split the data into positive and negative gamma exposure
        positive_data = all_data[all_data[self.y_col] >= 0]
        negative_data = all_data[all_data[self.y_col] < 0]

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Set background color
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)

        # Plot the data using both positive and negative gamma exposure
        ax.bar(positive_data[self.x_col], positive_data[self.y_col], color=positive_color, label='Positive Exposure',
               width=self.bar_width)
        ax.bar(negative_data[self.x_col], negative_data[self.y_col], color=negative_color, label='Negative Exposure',
               width=self.bar_width)

        # Plot the line data if provided
        if line_data:
            ax.plot(list(line_data.keys()), list(line_data.values()), color=line_color, label='Line Data', linewidth=2)

        # Set grid, axis, and title colors
        if self.grid:
            ax.grid(True, color=self.grid_color, linestyle=self.grid_line_style, linewidth=self.grid_line_width)
        ax.spines['top'].set_color(self.axis_color)
        ax.spines['bottom'].set_color(self.axis_color)
        ax.spines['left'].set_color(self.axis_color)
        ax.spines['right'].set_color(self.axis_color)
        ax.xaxis.label.set_color(self.label_color)
        ax.yaxis.label.set_color(self.label_color)
        ax.tick_params(axis='x', colors=self.tick_color)
        ax.tick_params(axis='y', colors=self.tick_color)
        ax.axhline(0, color='black', linewidth=1.5)  # Zero line

        # Set title and label sizes and colors
        ax.set_title(title, fontsize=self.title_font_size, color=self.title_font_color)
        plt.xlabel(self.x_col, fontsize=self.xlabel_font_size, color=self.label_color)
        plt.ylabel(self.y_col, fontsize=self.ylabel_font_size, color=self.label_color)

        plt.xticks(rotation=self.x_rotation)

        # Add legend if enabled
        if self.legend:
            plt.legend()

        # Add the fixed MarketQuant logo to the chart
        self.add_logo(ax)

        # Show the chart
        plt.tight_layout()
        plt.show()

    def confirm_valid_data(self):
        """
        Confirms that the data is in the correct format for plotting.
        """
        return self.validate_data()
