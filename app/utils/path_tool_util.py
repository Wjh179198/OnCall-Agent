"""
为整个工程提供统一的绝对路径
"""
import os.path


def get_project_root() -> str:
    """
    获取文件所在的根目录
    :return: 字符串根目录
    """

    # 获取当前文件的绝对路径
    current_file = os.path.abspath(__file__)
    # 获取文件的根目录 先获取文件所在的文件夹的绝对路径
    current_dir = os.path.dirname(current_file)
    # 获取工程的根目录
    return os.path.dirname(current_dir)


def get_abs_path(relative_path : str) -> str:
    """
    通过相对路径获取绝对路径
    :param relative_path:相对路径
    :return: 放回绝对路径
    """
    project_path = get_project_root()
    return os.path.join(project_path, relative_path)

if __name__ == "__main__":
    print(get_abs_path("config\\config.txt"))