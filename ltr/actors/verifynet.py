from . import BaseActor


class VerifyActor(BaseActor):
    def __call__(self, data):

        train_feat_embedding, test_feat_embedding = self.net(data['train_images'], data['test_images'])

        # Compute multiple triple loss
        loss1 = 0
        for embedding in train_feat_embedding[1:]:
            loss_tmp = self.objective(train_feat_embedding[0], test_feat_embedding[0], embedding)
            loss1 = loss1 + loss_tmp

        loss2 = 0
        for embedding in test_feat_embedding[1:]:
            loss_tmp = self.objective(test_feat_embedding[0], train_feat_embedding[0], embedding)
            loss2 = loss2 + loss_tmp

        loss = loss1 + loss2

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/tri1_loss': loss1.item(),
                 'Loss/tri2_loss': loss2.item()}

        return loss, stats
