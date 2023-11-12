class Configs:
    file_paths = ""
    joint_Position_tags = ""
    frames = ""
    hands = ""
    timestamp_usec = ""
    timestamp = ""
    distance = ""
    velocity = ""
    combined = ""
    output_extenstion = ""
    img_output_dir = ""
    is_pd = ""
    def __init__(self, file_paths, joint_position_tags, frames, hands, timestamp_usec, timestamp, distance, velocity,
                 combined,output_extenstion,img_output_dir,is_pd):
        self.file_paths = file_paths
        self.joint_Position_tags = joint_position_tags
        self.frames = frames
        self.hands = hands
        self.timestamp_usec = timestamp_usec
        self.timestamp = timestamp
        self.distance = distance
        self.velocity = velocity
        self.combined = combined
        self.output_extenstion = output_extenstion
        self.img_output_dir = img_output_dir
        self.is_pd = is_pd

    def get_Configs(self):
        return self

    def set_Configs(self, file_paths, joint_position_tags, frames, hands, timestamp_usec, timestamp, distance,
                    velocity, output_extenstion,img_output_dir,is_pd):
        self.file_paths = file_paths
        self.joint_Position_tags = joint_position_tags
        self.frames = frames
        self.hands = hands
        self.timestamp_usec = timestamp_usec
        self.timestamp = timestamp
        self.distance = distance
        self.velocity = velocity
        self.output_extenstion = output_extenstion
        self.img_output_dir = img_output_dir
        self.is_pd = is_pd
