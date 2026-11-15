"""Strategy E — word segmentation for PhoBERT (amendment v3.4).

PhoBERT was pre-trained on word-segmented text ("thủ_tục hủy đơn"), so its input must be segmented the same way.
Primary: RDRSegmenter of VnCoreNLP through py_vncorenlp (needs Java), as recommended by the PhoBERT authors.
Recorded fallback: underthesea.word_tokenize(format='text') when VnCoreNLP cannot start. `backend` says which one
ran; texts the segmenter fails on are kept unsegmented and counted in `failures`.
"""
import os

VNCORENLP_JAR = 'VnCoreNLP-1.2.jar'


class Segmenter:
    def __init__(self, backend='auto', save_dir=None, log=print):
        self.cache, self.failures, self.errors = {}, 0, []
        self.backend, self._fn = None, None
        if backend in ('auto', 'vncorenlp'):
            try:
                self._fn = self._vncorenlp(save_dir or os.path.abspath('vncorenlp'))
                self.backend = 'vncorenlp-rdrsegmenter'
            except Exception as e:                 # Java missing, download blocked, …
                self.errors.append(f'VnCoreNLP: {type(e).__name__}: {str(e)[:300]}')
                if backend == 'vncorenlp':
                    raise
        if self._fn is None:
            from underthesea import word_tokenize
            self._fn = lambda t: word_tokenize(t, format='text')
            self.backend = 'underthesea'
            if self.errors:
                log('Không dùng được VnCoreNLP, chuyển sang underthesea: ' + self.errors[0])

    @staticmethod
    def _vncorenlp(save_dir):
        import py_vncorenlp
        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        cwd = os.getcwd()
        try:
            if not os.path.exists(os.path.join(save_dir, VNCORENLP_JAR)):
                py_vncorenlp.download_model(save_dir=save_dir)
            seg = py_vncorenlp.VnCoreNLP(annotators=['wseg'], save_dir=save_dir)
        finally:
            os.chdir(cwd)                          # py_vncorenlp changes the working directory
        return lambda t: ' '.join(seg.word_segment(t))

    def __call__(self, texts):
        out = []
        for t in texts:
            t = str(t)
            if t not in self.cache:
                if not t.strip():
                    self.cache[t] = t
                else:
                    try:
                        self.cache[t] = self._fn(t)
                    except Exception:
                        self.cache[t] = t
                        self.failures += 1
            out.append(self.cache[t])
        return out
