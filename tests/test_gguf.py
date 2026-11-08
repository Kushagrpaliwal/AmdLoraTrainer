import sys
import pytest
from amdlora.io.gguf import convert_to_gguf, find_convert_script, find_quantize_command

FAKE_CONVERT = '''
import os, sys
args = sys.argv[1:]
model_dir, outfile, outtype = args[0], args[args.index("--outfile") + 1], args[args.index("--outtype") + 1]
assert os.path.isdir(model_dir)
open(outfile, "wb").write(b"FAKE-GGUF-" + outtype.encode())
open(outfile + ".pythonpath", "w").write(os.environ.get("PYTHONPATH", ""))
'''

FAKE_QUANTIZE = '''
import sys
src, dst, kind = sys.argv[1:4]
open(dst, "wb").write(open(src, "rb").read() + b"-QUANT-" + kind.encode())
'''

FAILING_CONVERT = "import sys; sys.exit(3)"


def make_fake_llama(tmp_path, convert_code=FAKE_CONVERT):
    llama = tmp_path / "llama.cpp"
    (llama / "gguf-py").mkdir(parents=True)
    (llama / "convert_hf_to_gguf.py").write_text(convert_code)
    quantize = tmp_path / "fake_quantize.py"
    quantize.write_text(FAKE_QUANTIZE)
    return llama, quantize


def make_model_dir(tmp_path):
    model = tmp_path / "merged"
    model.mkdir()
    (model / "config.json").write_text("{}")
    return model


def test_convert_then_quantize_writes_expected_files(tmp_path, monkeypatch):
    llama, quantize = make_fake_llama(tmp_path)
    # The wrapper runs "<quantize command> in out type"; use python + our fake script.
    monkeypatch.setattr("amdlora.io.gguf.find_quantize_command",
                        lambda llama_dir, explicit=None: [sys.executable, str(quantize)])
    out = tmp_path / "gguf"
    files = convert_to_gguf(make_model_dir(tmp_path), out, llama, name="demo", quants=("Q8_0", "Q4_K_M"))
    assert [f.name for f in files] == ["demo-bf16.gguf", "demo-Q8_0.gguf", "demo-Q4_K_M.gguf"]
    assert files[0].read_bytes() == b"FAKE-GGUF-bf16"
    assert files[1].read_bytes().endswith(b"-QUANT-Q8_0")
    assert str(llama / "gguf-py") in (out / "demo-bf16.gguf.pythonpath").read_text()


def test_missing_config_raises(tmp_path):
    llama, _ = make_fake_llama(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        convert_to_gguf(empty, tmp_path / "out", llama, quants=())


def test_missing_convert_script_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_convert_script(tmp_path)


def test_missing_quantize_tool_raises_before_converting(tmp_path, monkeypatch):
    llama, _ = make_fake_llama(tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.delenv("LLAMA_QUANTIZE", raising=False)
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        convert_to_gguf(make_model_dir(tmp_path), out, llama, quants=("Q4_K_M",))
    assert not out.exists()                                  # nothing was converted


def test_failing_conversion_raises_runtime_error(tmp_path):
    llama, _ = make_fake_llama(tmp_path, convert_code=FAILING_CONVERT)
    with pytest.raises(RuntimeError):
        convert_to_gguf(make_model_dir(tmp_path), tmp_path / "out", llama, quants=())


def test_explicit_quantize_path_is_used(tmp_path):
    exe = tmp_path / "my-quantize"
    exe.write_text("")
    assert find_quantize_command(tmp_path, explicit=exe) == [str(exe)]