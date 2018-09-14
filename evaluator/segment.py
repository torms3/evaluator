from __future__ import print_function


__all__ = ['Segment', 'Compose', 'Segmentation']


class Segment(object):
    """
    Abstract interface.
    """
    def __init__(self):
        pass

    def __call__(self, data, **kwargs):
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += ')'
        return format_string


class Compose(Segment):
    """Composes several segments together.

    Args:
        segments (list of ``Segment`` objects): list of segments to compose.
    """
    def __init__(self, segments):
        self.segments = segments

    def __call__(self, data, **kwargs):
        for seg in self.segments:
            data = seg(data, **kwargs)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for seg in self.segments:
            format_string += '\n'
            format_string += '    {}'.format(seg)
        format_string += '\n)'
        return format_string


class Segmentation(object):
    def __init__(self, data, segment=None):
        self.data = data
        self.segment = Segment() if segment is None else segment

    def __call__(self, **kwargs):
        return self.segment(self.data, **kwargs)
