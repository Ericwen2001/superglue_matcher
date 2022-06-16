import torch
from datasets.superpoint_dataset import SuperPointDataset
from models.superglue import SuperGlue
class SuperGlueMatcher():
    def __init__(self) -> None:

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = {
            'supgerglue': {
                'load_stat': 'True',
                'weights': 'superglue_indoor', #indoor/outdoor pretrained weight.
                'training': 'False',
                'sinkhorn_iterations': 20,
                'keypoint_encoder': [32, 64, 128, 256],
                'match_threshold': 0.2,
                'descriptor_dim': 256, #superpoint dimension
                'GNN_layers': ['self', 'cross'] * 9
        },
    }
        self.matcher = SuperGlue(self.config.get('supgerglue', {})) 
        if torch.cuda.is_available():
            self.matcher.cuda()
        else:
            print("### CUDA not available ###")
        
    
    def get_match(self, des0, des1, pts0, pts1,image0,image1,scores0,scores1):
        '''
        input params:
            @des0: descriptor for frame 0 
            @des1: descriptor for frame 1
            @pts0: keypoints for frame 0
            @pts1: keypoints for frame 1 
            @image0: frame 0 # normolization 
            @image1: frame 1 # normolization
            @scores0
            @scores1
        output:
            @idx0: index of matched keypoints in frame0
            @idx1: index of matched keypoints in frame1
        '''
        pred = {}
        
        if torch.cuda.is_available():
            pred['keypoints0'] = pts0.cuda()
            pred['keypoints1'] = pts1.cuda()
            pred['descriptors0'] = torch.stack(des0).cuda()
            pred['descriptors1'] = torch.stack(des1).cuda()
            pred['image0'] = image0
            pred['image1'] = image1
            pred['scores0'] = torch.stack(scores0).cuda()
            pred['scores1'] = torch.stack(scores1).cuda()
        else:
            pred['keypoints0'] = pts0
            pred['keypoints1'] = pts1
            pred['descriptors0'] = torch.stack(des0)
            pred['descriptors1'] = torch.stack(des1)
            pred['image0'] = image0
            pred['image1'] = image1
            pred['scores0'] = torch.stack(scores0)
            pred['scores1'] = torch.stack(scores1)
        
        data = self.matcher(pred)

        return data['matches0'], data['matches1']
        # -1 stands for unmatched


# # Example of use

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.2,
            'max_keypoints': 1024,
        }
}
train_set = SuperPointDataset("C:/datasets/coco2014/train2014", device=device, superpoint_config=config.get('superpoint', {}))
train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size=1, drop_last=True)
SuperGlueMatcher = SuperGlueMatcher()
for i, pred in enumerate(train_loader):
    image0 = pred['image0']
    image1 = pred['image1']
    descriptors0 = pred['descriptors0']
    descriptors1 = pred['descriptors1']
    keypoints0 = pred['keypoints0']
    keypoints1 = pred['keypoints1']
    sc0 = pred['scores0']
    sc1 = pred['scores1']
    res = SuperGlueMatcher.get_match(des0=descriptors0,des1=descriptors1,image0=image0,image1=image1,
    pts0=keypoints0,pts1=keypoints1,scores0=sc0,scores1=sc1)
    print(res)
    