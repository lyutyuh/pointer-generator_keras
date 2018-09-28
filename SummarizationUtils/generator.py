import numpy as np


class SummarizationGenerator():
    def __init__(self, hps, batcher, just_enc=False):
        self.hps = hps
        self.batcher = batcher
        self.just_enc = just_enc

    def make_feed_dict(self, batch):
        """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.
        Args:
          batch: Batch object
          just_enc: Boolean. If True, only feed the parts needed for the encoder.
        """
        sequence_of_inputs = ['enc_batch', 'enc_lens', 'enc_padding_mask',
                              'enc_batch_extend_vocab', 'max_art_oovs_inp',
                              'dec_batch', 'target_batch', 'dec_padding_mask']

        feed_dict = {}
        feed_dict['enc_batch'] = batch.enc_batch
        feed_dict['enc_lens'] = batch.enc_lens
        feed_dict['enc_padding_mask'] = batch.enc_padding_mask
        if self.hps.pointer_gen:
            feed_dict['enc_batch_extend_vocab'] = batch.enc_batch_extend_vocab
            feed_dict['max_art_oovs_inp'] = batch.max_art_oovs ### sth is wrong
        if not self.just_enc:
            feed_dict['dec_batch'] = batch.dec_batch
            feed_dict['target_batch'] = batch.target_batch
            feed_dict['dec_padding_mask'] = batch.dec_padding_mask

        targets = [
            np.zeros(shape=(self.hps.batch_size, self.hps.max_dec_steps, self.hps.vocab_size + batch.max_art_oovs[0])),
            np.tile(np.expand_dims(batch.target_batch, axis=-1), [1, 1, self.hps.batch_size])] if self.hps.use_coverage else [
            np.tile(np.expand_dims(batch.target_batch, axis=-1), [1, 1, self.hps.batch_size])]
        # Return:
        # X, y
        return [np.array(feed_dict[x]) for x in sequence_of_inputs], targets

    def get_batch_generator(self):
        a_batch = self.batcher.next_batch()
        while a_batch is not None:
            yield self.make_feed_dict(a_batch)
            a_batch = self.batcher.next_batch()

