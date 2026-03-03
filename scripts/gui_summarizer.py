import os
import sys

# Fix for OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tkinter as tk
# Note: A new, beautified web interface is available at scripts/web_ui.py
# Run it with: streamlit run scripts/web_ui.py
from tkinter import ttk, scrolledtext, filedialog, messagebox
from tkinter import font as tkfont
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.inference import T5Summarizer
from configs.config import MODEL_NAME, MODEL_CACHE_DIR, INFERENCE_CONFIG
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import importlib.util


class SummarizerGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("英文技术文档摘要生成（原版/微调可选）")
        self.master.geometry("1020x700")
        self.master.minsize(860, 600)

        self.model_type = tk.StringVar(value="pretrained")
        self.model_dir = tk.StringVar(value=os.path.join("outputs", "models", "best_model"))
        self.max_length = tk.IntVar(value=INFERENCE_CONFIG["max_length"]) 
        self.num_beams = tk.IntVar(value=INFERENCE_CONFIG["num_beams"]) 
        self.translate_source = tk.StringVar(value="摘要")
        self.translate_direction = tk.StringVar(value="英->中")
        self._translator_cache = {}

        top = ttk.Frame(master)
        top.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top, text="模型选择:").pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="原版", variable=self.model_type, value="pretrained", command=self._toggle_path).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(top, text="微调", variable=self.model_type, value="fine_tuned", command=self._toggle_path).pack(side=tk.LEFT, padx=5)

        path_frame = ttk.Frame(master)
        path_frame.pack(fill=tk.X, padx=10)
        ttk.Label(path_frame, text="微调模型目录:").pack(side=tk.LEFT)
        self.path_entry = ttk.Entry(path_frame, textvariable=self.model_dir, width=60)
        self.path_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(path_frame, text="浏览", command=self._browse_dir).pack(side=tk.LEFT)

        params = ttk.Frame(master)
        params.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(params, text="max_length").pack(side=tk.LEFT)
        ttk.Spinbox(params, from_=50, to=512, increment=10, textvariable=self.max_length, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(params, text="num_beams").pack(side=tk.LEFT, padx=10)
        ttk.Spinbox(params, from_=1, to=10, increment=1, textvariable=self.num_beams, width=6).pack(side=tk.LEFT, padx=5)

        self.io = ttk.PanedWindow(master, orient=tk.HORIZONTAL)
        self.io.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 60))

        left = ttk.Frame(self.io)
        right = ttk.Frame(self.io)
        self.io.add(left, weight=1)
        self.io.add(right, weight=1)

        self.left_split = ttk.PanedWindow(left, orient=tk.VERTICAL)
        self.left_split.pack(fill=tk.BOTH, expand=True)
        left_top = ttk.Frame(self.left_split)
        left_bottom = ttk.Frame(self.left_split)
        self.left_split.add(left_top, weight=1)
        self.left_split.add(left_bottom, weight=1)
        ttk.Label(left_top, text="输入英文技术文档").pack(anchor=tk.W)
        self.input_text = scrolledtext.ScrolledText(left_top, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True)
        ttk.Label(left_bottom, text="原文翻译").pack(anchor=tk.W)
        self.translation_input_text = scrolledtext.ScrolledText(left_bottom, wrap=tk.WORD)
        self.translation_input_text.pack(fill=tk.BOTH, expand=True)

        self.right_split = ttk.PanedWindow(right, orient=tk.VERTICAL)
        self.right_split.pack(fill=tk.BOTH, expand=True)
        right_top = ttk.Frame(self.right_split)
        right_bottom = ttk.Frame(self.right_split)
        self.right_split.add(right_top, weight=1)
        self.right_split.add(right_bottom, weight=1)
        ttk.Label(right_top, text="生成摘要").pack(anchor=tk.W)
        self.output_text = scrolledtext.ScrolledText(right_top, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        ttk.Label(right_bottom, text="摘要翻译").pack(anchor=tk.W)
        self.translation_summary_text = scrolledtext.ScrolledText(right_bottom, wrap=tk.WORD)
        self.translation_summary_text.pack(fill=tk.BOTH, expand=True)
        txt_font = tkfont.Font(family="Microsoft YaHei UI", size=12)
        self.input_text.configure(font=txt_font)
        self.translation_input_text.configure(font=txt_font)
        self.output_text.configure(font=txt_font)
        self.translation_summary_text.configure(font=txt_font)

        actions = ttk.Frame(master)
        actions.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5, before=self.io)
        ttk.Button(actions, text="生成摘要", command=self._generate).pack(side=tk.LEFT)
        ttk.Label(actions, text="方向").pack(side=tk.LEFT, padx=10)
        ttk.Combobox(actions, textvariable=self.translate_direction, values=["英->中", "中->英"], state="readonly", width=8).pack(side=tk.LEFT)
        ttk.Button(actions, text="翻译", command=self._translate).pack(side=tk.LEFT, padx=10)
        try:
            actions.lift()
        except Exception:
            pass

        self._toggle_path()
        self.master.update_idletasks()
        self._fit_window()
        self._set_initial_layout()
        self.master.after(100, self._set_initial_layout)

    def _toggle_path(self):
        state = tk.NORMAL if self.model_type.get() == "fine_tuned" else tk.DISABLED
        self.path_entry.configure(state=state)

    def _browse_dir(self):
        d = filedialog.askdirectory(initialdir=self.model_dir.get() or os.getcwd())
        if d:
            self.model_dir.set(d)

    def _generate(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请输入英文技术文档文本")
            return
        model_path = None
        if self.model_type.get() == "fine_tuned":
            p = self.model_dir.get().strip()
            if not p or not os.path.exists(p):
                messagebox.showerror("错误", "微调模型目录不存在")
                return
            model_path = p
        try:
            start = time.time()
            summarizer = T5Summarizer(model_path=model_path, tokenizer_name=MODEL_NAME)
            summary = summarizer.generate_summary(
                text,
                max_length=int(self.max_length.get()),
                num_beams=int(self.num_beams.get()),
                min_length=50,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=2.0,
                max_source_length=512,
            )
            elapsed = time.time() - start
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, summary + f"\n\n耗时: {elapsed:.2f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"{e}\n\nCheck console for details.")

    def _get_translator(self, direction):
        if direction == "英->中":
            model_name = "Helsinki-NLP/opus-mt-en-zh"
            task = "translation"
        else:
            model_name = "Helsinki-NLP/opus-mt-zh-en"
            task = "translation"
        if model_name in self._translator_cache:
            return self._translator_cache[model_name]
        if importlib.util.find_spec("sentencepiece") is None:
            raise RuntimeError("缺少 sentencepiece 依赖，请先安装：pip install sentencepiece")
        local_dir = os.path.join(MODEL_CACHE_DIR, model_name.replace("/", os.sep))
        try:
            if os.path.isdir(local_dir):
                tok = AutoTokenizer.from_pretrained(local_dir, use_fast=False, local_files_only=True)
                try:
                    mdl = AutoModelForSeq2SeqLM.from_pretrained(local_dir, local_files_only=True)
                except Exception:
                    mdl = AutoModelForSeq2SeqLM.from_pretrained(local_dir, local_files_only=True, use_safetensors=False)
            else:
                tok = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR, use_fast=False, local_files_only=True)
                try:
                    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR, local_files_only=True)
                except Exception:
                    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR, local_files_only=True, use_safetensors=False)
        except Exception as e:
            raise RuntimeError(f"加载翻译模型失败: {e}\n请在联网环境运行 scripts\\download_assets.py 以一次性下载到 {MODEL_CACHE_DIR}。")
        nlp = pipeline(task, model=mdl, tokenizer=tok)
        self._translator_cache[model_name] = nlp
        return nlp

    def _translate(self):
        in_text = self.input_text.get("1.0", tk.END).strip()
        sum_text = self.output_text.get("1.0", tk.END).strip()
        if "\n\n耗时:" in sum_text:
            sum_text = sum_text.split("\n\n耗时:")[0].strip()
        if not in_text and not sum_text:
            messagebox.showwarning("提示", "没有可翻译的文本")
            return
        try:
            tr = self._get_translator(self.translate_direction.get())
            if in_text:
                res_in = tr(in_text)
                out_in = " ".join([r.get("translation_text", "") for r in res_in]) if isinstance(res_in, list) else str(res_in)
                self.translation_input_text.delete("1.0", tk.END)
                self.translation_input_text.insert(tk.END, out_in.strip())
            else:
                self.translation_input_text.delete("1.0", tk.END)
            if sum_text:
                res_sum = tr(sum_text)
                out_sum = " ".join([r.get("translation_text", "") for r in res_sum]) if isinstance(res_sum, list) else str(res_sum)
                self.translation_summary_text.delete("1.0", tk.END)
                self.translation_summary_text.insert(tk.END, out_sum.strip())
            else:
                self.translation_summary_text.delete("1.0", tk.END)
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"{e}\n\nCheck console for details.")

    def _fit_window(self):
        try:
            self.master.update_idletasks()
            screen_w = self.master.winfo_screenwidth()
            screen_h = self.master.winfo_screenheight()
            w = min(1100, int(screen_w * 0.7))
            h = min(740, int(screen_h * 0.7))
            self.master.geometry(f"{w}x{h}")
        except Exception:
            pass

    def _set_initial_layout(self):
        try:
            self.master.update_idletasks()
            w = self.io.winfo_width()
            hl = self.left_split.winfo_height()
            hr = self.right_split.winfo_height()
            if w < 10 or hl < 10 or hr < 10:
                self.master.after(100, self._set_initial_layout)
                return
            self.io.sashpos(0, w // 2)
            self.left_split.sashpos(0, hl // 2)
            self.right_split.sashpos(0, hr // 2)
        except Exception:
            pass


def main():
    root = tk.Tk()
    SummarizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

