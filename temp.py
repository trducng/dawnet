import difflib
import ipywidgets as widgets

from IPython.display import display
text_src = widgets.Textarea(placeholder="Source")
text_des = widgets.Textarea(placeholder="Destination")
button = widgets.Button(description="Submit")

output = widgets.HTML()

display(text_src, text_des, button, output)

def on_button_clicked(b):
    html_diff = difflib.HtmlDiff().make_file(text_src.value.splitlines(), text_des.value.splitlines())
    html_diff = html_diff.replace("text-align:right", "text-align:left")
    html_diff = html_diff.replace("<td", "<td style='text-align:left'")

    html_diff = html_diff.replace("background-color:#aaffaa", "background-color:#aaffaa;color: black")
    html_diff = html_diff.replace("background-color:#ffff77", "background-color:#ffff77;color: black")
    html_diff = html_diff.replace("background-color:#ffaaaa", "background-color:#ffaaaa;color: black")
    output.value = html_diff

button.on_click(on_button_clicked)
