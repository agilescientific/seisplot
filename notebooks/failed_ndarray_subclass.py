# Failed attempt at Seismic subclass of ndarray
import numpy as np

class Seismic(np.ndarray):
    """
    A fancy ndarray. Gives some utility functions, plotting, etc,
    for seismic data.
    """
    def __new__(cls, data, params=None, dtype=float):
        """
        Handles what happens when you do Seismic(). Following
        https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#
        slightly-more-realistic-example-attribute-added-to-existing-array
        """
        obj = np.asarray(data, dtype=dtype).view(cls)
        obj.params = params
        return obj

    def __array_finalize__(self, obj):
        """
        This method sees the creation of all Seismic objects, e.g. those
        created by slicing etc. OTOH, __new__ only sees those created by
        an explicit Seismic() call. So all the logic goes here.
        """
        if obj is None:
            return

        params = getattr(obj, 'params', {})
        for k, v in params.items():
            print("Setting", k)
            setattr(self, k, v)

        self.inlines = getattr(obj, 'inlines', 1)
        self.xlines = getattr(obj, 'xlines', 0)
        self.cips = np.product(self.shape[:-1])
        self.tsamples = self.shape[-1]
        
        print(self.inlines)
        if self.inlines > 1:
            # Then it's a 3D
            print("3D")
            x = self.xlines
            self.xlines = int(self.shape[0] / self.inlines)
            if x != self.xlines:
                s = "Number of xlines changed to {} to match apparent survey size."
                print(s.format(self.xlines))

        return
