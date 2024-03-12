import os, sys
sys.path.append('./')

import argparse
import gradio as gr
from inference.pipeline import infer_from_text, infer_from_wav
import random
import time

class Inferer():
    def infer_once_args(self, *args, **kargs):
        assert len(kargs) == 0
        keys = [
            'input_audio_name',
            'input_text_box',
        ]
        inp = {}
        out_name = None
        info = ""
        
        print(f"args {args}")
        try: # try to catch errors and jump to return 
            for key_index in range(len(keys)):
                key = keys[key_index]
                inp[key] = args[key_index]
                if '_name' in key or '_ckpt' in key:
                    inp[key] = inp[key] if inp[key] is not None else ''
            
            inp['input_text_box'] = inp['input_text_box'].strip()

            if inp['input_audio_name']=='' and inp['input_text_box']=='':
                info = "Input Error: Input audio OR text is REQUIRED!"
                raise ValueError
                
            inp['out_name'] = f"temp/out_{generate_random_uuid()}.mp4"
            
            print(f"infer inputs : {inp}")
            
            try:
                if inp['input_text_box']=='':
                    output_text, out_name, asr_text = infer_from_wav(inp['input_audio_name'], video_path=inp['out_name'])
                else:
                    output_text, out_name, _ = infer_from_text(inp['input_text_box'], video_path=inp['out_name'])
                    asr_text = ''
            except Exception as e:
                content = f"{e}"
                info = f"Inference ERROR: {content}"
        except Exception as e:
            if info == "": # unexpected errors
                content = f"{e}"
                info = f"WebUI ERROR: {content}"
        
        # output part
        
        if len(info) > 0 : # there is errors    
            print(info)
            info_gr = gr.update(visible=True, value=info)
        else: # no errors
            info_gr = gr.update(visible=False, value=info)
            
        if out_name is not None and len(out_name) > 0 and os.path.exists(out_name): # good output
            print(f"Succefully generated in {out_name}")
            video_gr = gr.update(visible=True, value=out_name)
        else:
            print(f"Failed to generate")
            video_gr = gr.update(visible=True, value=out_name)
            
        return asr_text, output_text, video_gr, info_gr

def toggle_audio_file(choice):
    if choice == False:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
    
def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)

# generate random uuid and do not disturb global random state
def generate_random_uuid(len_uuid = 16):
    prev_state = random.getstate()
    random.seed(time.time())
    s = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    res = ''.join(random.choices(s, k=len_uuid))
    random.setstate(prev_state)
    return res

def genefacepp_demo(
    device          = None,
    warpfn          = None,
    ):

    sep_line = "-" * 40

    infer_obj = Inferer()

    print(sep_line)
    print("Model loading is finished.")
    print(sep_line)
    with gr.Blocks(analytics_enabled=False) as genefacepp_interface:
        gr.Markdown("\
            <div align='center'> <h1> ChatFace Demo </span> </h1> \
            内部演示，请勿转发 </div>")
        
        sources = None
        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="driven_audio"):
                    with gr.TabItem('对话输入'):
                        gr.Markdown("语音输入 或 文本输入，二选一")
                        with gr.Column(variant='panel'):
                            input_audio_name = gr.Audio(label="输入对话语音", sources=sources, type="filepath", value='infer_outs/tts_output.wav')
                            input_text_box = gr.Textbox(label="输入对话文本：", value="", interactive=True, visible=True, show_label=True)
                            submit = gr.Button('生成', elem_id="generate", variant='primary')

            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="checkbox"):
                    with gr.TabItem('对话 LLM 数字人'):
                        with gr.Column(variant='panel'):
                            asr_output_box = gr.Textbox(label="语音转换", value="", interactive=False, visible=True, show_label=True, lines=1)
                            llm_output_box = gr.Textbox(label="文本生成", value="", interactive=False, visible=True, show_label=True, lines=5)

                    with gr.Tabs(elem_id="genearted_video"):
                        info_box = gr.Textbox(label="Error", interactive=False, visible=False)
                        gen_video = gr.Video(label="视频生成", format="mp4", visible=True, autoplay=True)

            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="checkbox"):
                    with gr.TabItem('General Settings'):
                        with gr.Column(variant='panel'):
                            gr.Markdown("暂无")


        fn = infer_obj.infer_once_args
        if warpfn:
            fn = warpfn(fn)
        submit.click(
                    fn=fn, 
                    inputs=[ 
                        input_audio_name,
                        input_text_box,
                    ], 
                    outputs=[
                        asr_output_box,
                        llm_output_box,                    
                        gen_video,
                        info_box,
                    ],
                    )

    print(sep_line)
    print("Gradio page is constructed.")
    print(sep_line)

    return genefacepp_interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None) 
    parser.add_argument("--server", type=str, default='127.0.0.1') 

    args = parser.parse_args()
    demo = genefacepp_demo(
        #device='cuda',
        warpfn=None,
    )
    demo.queue()
    demo.launch(server_name=args.server, server_port=args.port)


