import re
import os 
import platform
import socket
import subprocess
from datetime import datetime

import distro
import psutil
from screeninfo import get_monitors
from rich import box
from rich.console import Console
from rich.table import Table
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

def get_uptime():

    boot_time = psutil.boot_time()
    now = datetime.now().timestamp()

    uptime_seconds = int(now - boot_time)
    days, remainder = divmod(uptime_seconds, 86400) # 60 sec/min * 60 min/hr * 24 hr/day
    hours, remainder = divmod(remainder, 3600) # 60 sec/min * 60 min/hr
    minutes, _ = divmod(remainder, 60)

    uptime_str = f"{days} days, {hours} hours, {minutes} minutes"
    return uptime_str

# print(get_uptime())

def get_shell():
    
    shell = os.environ.get("COMSPEC")
    return os.path.basename(shell) if shell else "Unknown"

# print(get_shell())

def get_de():

    de = os.environ.get("SESSIONNAME")
    return de if de else "Unknown"

# print(get_de())

def get_resolution():

    monitors = get_monitors()
    resolutions = [f"{monitor.width}x{monitor.height}" for monitor in monitors]
    return ", ".join(resolutions)

# print(get_resolution())

def get_packages():

    try:
        result = subprocess.check_output(["powershell", "Get-AppxPackage"], stderr=subprocess.PIPE, shell=True, text=True)
        packages = len([line for line in result.splitlines() if line])
        return str(packages)
    except subprocess.CalledProcessError as e:
        return str(e)
    
# print(get_packages())

def get_os():

    os_name = platform.system()
    os_release = platform.release()
    os_version = platform.version()
    return f"{os_name} {os_release} {os_version}"

# print(get_os())

def get_gpu():

    try:
        result = subprocess.check_output(["wmic", "path", "win32_VideoController", "get", "name"], stderr=subprocess.PIPE, shell=True, text=True)
        gpu = [line for line in result.splitlines() if line][1]
        return gpu
    except subprocess.CalledProcessError as e:
        return str(e)
    
# print(get_gpu())

def get_cpu():
    return platform.processor()

# print(get_cpu())

def get_ram():

    ram = psutil.virtual_memory().total
    ram = ram / 1024**3
    return f"{ram:.1f} GiB"

# print(get_ram())

def get_kernel():
    
    kernel = platform.version()
    return kernel

def get_host():

    return socket.gethostname()



if __name__ == '__main__':
    console = Console()

    # Gather system information
    os_info = get_os()
    host = get_host()
    kernel = get_kernel()
    uptime = get_uptime()
    packages = get_packages()
    shell = get_shell()
    resolution = get_resolution()
    de = get_de()
    gpu = get_gpu()
    cpu = get_cpu()
    ram = get_ram()

    # Create the system information table
    table = Table(title="🚀 System Information 🚀", box=box.SIMPLE_HEAVY, title_style="bold magenta")
    label_color = "bold cyan"
    value_color = "bold white"

    info = [
        ("🖥️ OS", os_info),
        ("🔒 Host", host),
        ("🧬 Kernel", kernel),
        ("⏳ Uptime", uptime),
        ("📦 Packages", packages),
        ("💻 Shell", shell),
        ("📺 Resolution", resolution),
        ("🎨 DE", de),
        ("🔥 GPU", gpu),
        ("🧠 CPU", cpu),
        ("🔋 RAM", ram),
    ]

    for label, value in info:
        table.add_row(Text(f"{label}:", style=label_color), Text(value, style=value_color))

    # Create the styled and properly aligned logo
    logo_lines = [
        "########################################",
        "#                                      #",
        "#              [bold magenta]BAD[/bold magenta][bold blue]_AT_[/bold blue][bold green]AI[/bold green]               #",
        "#                                      #",
        "########################################"
    ]

    logo_text = "\n".join(logo_lines)  # Ensure alignment
    logo_panel = Panel(
        Text.from_markup(logo_text),
        border_style="bold magenta",
        box=box.HEAVY,
        padding=(1, 2),  # Add padding for more space around the logo
    )

    # Layout and display
    separator = Text("─" * 40, style="dim")  # Section separator

    # Center-align the logo and the table
    console.print(Align.center(logo_panel, vertical="middle"))
    console.print(Align.center(separator, vertical="middle"))
    console.print(Align.center(table, vertical="middle"))
