import gradio as gr

def test_fn(x):
    return x

with gr.Blocks() as demo:
    dropdown = gr.Dropdown(choices=["Option 1", "Option 2"], value="Option 1", label="Select Option")
    demo_button = gr.Button("Submit")
    output = gr.Textbox(label="Output")
    demo_button.click(test_fn, inputs=dropdown, outputs=output)
demo.launch()
