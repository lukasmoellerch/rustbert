import io
import json
import torch
from transformers import BertModel, BertTokenizer


tokenizer = BertTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-TinyBERT-L6-v2"
)
model = BertModel.from_pretrained("sentence-transformers/paraphrase-TinyBERT-L6-v2")

params = list(model.named_parameters()) # type: ignore

with open("tinybert.bin", "wb") as f:
    # write the number of parameters as u64
    f.write(len(params).to_bytes(8, byteorder="little"))

    buf = io.BytesIO()

    config_bytes = json.dumps(model.config.to_dict()).encode("utf-8")  # type: ignore
    buf.write(len(config_bytes).to_bytes(8, byteorder="little"))
    buf.write(config_bytes)

    for name, param in params:
        # write name of the parameter and shape of the tensor as u64
        name_bytes = name.encode("utf-8")
        buf.write(len(name_bytes).to_bytes(8, byteorder="little"))
        buf.write(name_bytes)

        # write the type of the tensor as u8
        if param.dtype == torch.float32:
            buf.write(b"\x00")
        elif param.dtype == torch.float64:
            buf.write(b"\x01")
        else:
            raise ValueError("Unsupported dtype")

        shape = param.shape
        buf.write(len(shape).to_bytes(8, byteorder="little"))
        for dim in shape:
            buf.write(dim.to_bytes(8, byteorder="little"))
        
    # Write header size
    f.write(len(buf.getvalue()).to_bytes(8, byteorder="little"))
    
    # Write header
    f.write(buf.getvalue())
        
    # write the tensor data
    for name, param in params:
        print(name, f.tell())
        f.write(param.detach().numpy().tobytes())
    