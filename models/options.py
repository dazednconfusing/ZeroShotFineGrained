
class Options():
    def __init__(self, img_dim=None, txt_dim=None, proto_count=None, txt_to_img_hidden1=None,
                 txt_to_img_norm=None, apply_proto_dim_feed_forward=None, apply_proto_combine=None):
        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.proto_count = proto_count
        self.txt_to_img_hidden1 = txt_to_img_hidden1
        self.txt_to_img_norm = txt_to_img_norm
        self.apply_proto_dim_feed_forward = apply_proto_dim_feed_forward
        self.apply_proto_combine = apply_proto_combine
