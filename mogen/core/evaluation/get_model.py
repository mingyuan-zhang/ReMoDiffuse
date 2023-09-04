from mogen.models import build_submodule


def get_motion_model(name, ckpt_path):
    if name == 'kit_ml':
        model = build_submodule(dict(
            type='T2MMotionEncoder',
            input_size=251,
            movement_hidden_size=512,
            movement_latent_size=512,
            motion_hidden_size=1024,
            motion_latent_size=512,
        ))
    else:
        model = build_submodule(dict(
            type='T2MMotionEncoder',
            input_size=263,
            movement_hidden_size=512,
            movement_latent_size=512,
            motion_hidden_size=1024,
            motion_latent_size=512,
        ))
    model.load_pretrained(ckpt_path)
    return model

def get_text_model(name, ckpt_path):
    if name == 'kit_ml':
        model = build_submodule(dict(
            type='T2MTextEncoder',
            word_size=300,
            pos_size=15,
            hidden_size=512,
            output_size=512,
            max_text_len=20
        ))
    else:
        model = build_submodule(dict(
            type='T2MTextEncoder',
            word_size=300,
            pos_size=15,
            hidden_size=512,
            output_size=512,
            max_text_len=20
        ))
    model.load_pretrained(ckpt_path)
    return model
