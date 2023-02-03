"""Morphology Configuration"""
import pathlib


class MorphologyConfiguration:
    """Morphology configuration"""

    def __init__(
        self,
        name=None,
        path=None,
        format=None,
        seclist_names=None,
        secarray_names=None,
        section_index=None,
    ):
        """Init.

        Args:
            name (str): name of the morphology. If None, the file name from the path will
                be used.
            path (str): path to the morphology file.
            format (str): format of the morphology file, as to be 'asc' or 'swc'. If None,
                the extension of the path will be used.
            seclist_names (list): Optional. Names of the lists of sections
                (e.g: ['somatic', ...]).
            secarray_names (list): Optional. Names of the sections (e.g: ['soma', ...]).
            section_index (list): Optional. Index to a specific section, used for
                non-somatic recordings.
        """

        self.path = None
        if path:
            self.path = str(path)

        if name:
            self.name = name
        elif path:
            self.name = pathlib.Path(self.path).stem
        else:
            raise TypeError("name or path has to be informed")

        self.format = None
        if format:
            if self.path:
                if format.lower() != path[-3:].lower():
                    raise ValueError("The format does not match the morphology file")
            self.format = format
        elif self.path:
            self.format = path[-3:]

        if self.format and self.format.lower() not in ["asc", "swc"]:
            raise ValueError("The format of the morphology has to be 'asc' or 'swc'.")

        self.seclist_names = seclist_names
        self.secarray_names = secarray_names
        self.section_index = section_index

    def as_dict(self):
        return {
            "name": self.name,
            "format": self.format,
            "path": self.path,
            "seclist_names": self.seclist_names,
            "secarray_names": self.secarray_names,
            "section_index": self.section_index,
        }
