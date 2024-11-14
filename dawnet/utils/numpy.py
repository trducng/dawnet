"""From: https://github.com/xor2k/npy-append-array/tree/master
"""
import os, tempfile, threading
import warnings
from io import BytesIO, SEEK_END, SEEK_SET
from math import prod, ceil

import numpy
from numpy.compat import  isfileobj, pickle
from numpy.lib import format
from numpy.lib.format import _check_version, header_data_from_array_1_0


GROWTH_AXIS_MAX_DIGITS = 21  # = len(str(8*2**64-1)) hypothetical int1 dtype

def _wrap_header(header, version, header_len=None):
    """
    Takes a stringified header, and attaches the prefix and padding to it
    """
    import struct
    assert version is not None
    fmt, encoding = format._header_size_info[version]
    header = header.encode(encoding)
    hlen = len(header) + 1
    padlen = format.ARRAY_ALIGN - ((
        format.MAGIC_LEN + struct.calcsize(fmt) + hlen
    ) % format.ARRAY_ALIGN)
    try:
        header_prefix = format.magic(*version) + struct.pack(
            fmt, hlen + padlen
        )
    except struct.error:
        msg = "Header length {} too big for version={}".format(hlen, version)
        raise ValueError(msg) from None

    # Pad the header with spaces and a final newline such that the magic
    # string, the header-length short and the header are aligned on a
    # ARRAY_ALIGN byte boundary. This supports memory mapping of dtypes
    # aligned up to ARRAY_ALIGN on systems like Linux where mmap()
    # offset must be page-aligned (i.e. the beginning of the file).
    header = header_prefix + header + b' '*padlen 
    
    if header_len is not None:
        actual_header_len = len(header) + 1
        if actual_header_len > header_len:
            msg = (
                "Header length {} too big for specified header "+
                "length {}, version={}"
            ).format(actual_header_len, header_len, version)
            raise ValueError(msg) from None
        header += b' '*(header_len - actual_header_len)
    
    return header + b'\n'


def _wrap_header_guess_version(header, header_len=None):
    """
    Like `_wrap_header`, but chooses an appropriate version given the contents
    """
    try:
        return _wrap_header(header, (1, 0), header_len)
    except ValueError:
        pass

    try:
        ret = _wrap_header(header, (2, 0), header_len)
    except UnicodeEncodeError:
        pass
    else:
        warnings.warn("Stored array in format 2.0. It can only be"
                      "read by NumPy >= 1.9", UserWarning, stacklevel=2)
        return ret

    header = _wrap_header(header, (3, 0))
    warnings.warn("Stored array in format 3.0. It can only be "
                  "read by NumPy >= 1.17", UserWarning, stacklevel=2)
    return header


def _write_array_header(fp, d, version=None, header_len=None):
    """ Write the header for an array and returns the version used

    Parameters
    ----------
    fp : filelike object
    d : dict
        This has the appropriate entries for writing its string representation
        to the header of the file.
    version : tuple or None
        None means use oldest that works. Providing an explicit version will
        raise a ValueError if the format does not allow saving this data.
        Default: None
    header_len : int or None
        If not None, pads the header to the specified value or raises a
        ValueError if the header content is too big.
        Default: None
    """
    header = ["{"]
    for key, value in sorted(d.items()):
        # Need to use repr here, since we eval these when reading
        header.append("'%s': %s, " % (key, repr(value)))
    header.append("}")
    header = "".join(header)
    
    # Add some spare space so that the array header can be modified in-place
    # when changing the array size, e.g. when growing it by appending data at
    # the end. 
    shape = d['shape']
    header += " " * ((GROWTH_AXIS_MAX_DIGITS - len(repr(
        shape[-1 if d['fortran_order'] else 0]
    ))) if len(shape) > 0 else 0)
    
    if version is None:
        header = _wrap_header_guess_version(header, header_len)
    else:
        header = _wrap_header(header, version, header_len)
    fp.write(header)

def write_array(fp, array, version=None, allow_pickle=True, pickle_kwargs=None):
    """
    Write an array to an NPY file, including a header.
    If the array is neither C-contiguous nor Fortran-contiguous AND the
    file_like object is not a real file object, this function will have to
    copy data in memory.
    Parameters
    ----------
    fp : file_like object
        An open, writable file object, or similar object with a
        ``.write()`` method.
    array : ndarray
        The array to write to disk.
    version : (int, int) or None, optional
        The version number of the format. None means use the oldest
        supported version that is able to store the data.  Default: None
    allow_pickle : bool, optional
        Whether to allow writing pickled data. Default: True
    pickle_kwargs : dict, optional
        Additional keyword arguments to pass to pickle.dump, excluding
        'protocol'. These are only useful when pickling objects in object
        arrays on Python 3 to Python 2 compatible format.
    Raises
    ------
    ValueError
        If the array cannot be persisted. This includes the case of
        allow_pickle=False and array being an object array.
    Various other errors
        If the array contains Python objects as part of its dtype, the
        process of pickling them may raise various errors if the objects
        are not picklable.
    """
    _check_version(version)
    _write_array_header(fp, header_data_from_array_1_0(array), version)

    if array.itemsize == 0:
        buffersize = 0
    else:
        # Set buffer size to 16 MiB to hide the Python loop overhead.
        buffersize = max(16 * 1024 ** 2 // array.itemsize, 1)

    if array.dtype.hasobject:
        # We contain Python objects so we cannot write out the data
        # directly.  Instead, we will pickle it out
        if not allow_pickle:
            raise ValueError("Object arrays cannot be saved when "
                             "allow_pickle=False")
        if pickle_kwargs is None:
            pickle_kwargs = {}
        pickle.dump(array, fp, protocol=3, **pickle_kwargs)
    elif array.flags.f_contiguous and not array.flags.c_contiguous:
        if isfileobj(fp):
            array.T.tofile(fp)
        else:
            for chunk in numpy.nditer(
                    array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                    buffersize=buffersize, order='F'):
                fp.write(chunk.tobytes('C'))
    else:
        if isfileobj(fp):
            array.tofile(fp)
        else:
            for chunk in numpy.nditer(
                    array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                    buffersize=buffersize, order='C'):
                fp.write(chunk.tobytes('C'))

class _HeaderInfo():
    def __init__(self, fp):
        version = format.read_magic(fp)
        shape, fortran_order, dtype = format._read_array_header(fp, version)
        self.shape, self.fortran_order, self.dtype = (
            shape, fortran_order, dtype
        )

        header_size = fp.tell()
        self.header_size = header_size

        new_header = BytesIO()
        _write_array_header(new_header, {
            "shape": shape,
            "fortran_order": fortran_order,
            "descr": format.dtype_to_descr(dtype)
        })
        self.new_header = new_header.getvalue()

        fp.seek(0, SEEK_END)
        self.data_length = fp.tell() - header_size

        self.is_appendable = len(self.new_header) <= header_size

        self.needs_recovery = not (
            dtype.hasobject or
            self.data_length == prod(shape) * dtype.itemsize
        )

def is_appendable(filename):
    with open(filename, mode="rb") as fp:
        return _HeaderInfo(fp).is_appendable

def needs_recovery(filename):
    with open(filename, mode="rb") as fp:
        return _HeaderInfo(fp).needs_recovery

def ensure_appendable(filename, inplace=False):
    with open(filename, mode="rb+") as fp:
        hi = _HeaderInfo(fp)

        new_header_size = len(hi.new_header)

        if hi.is_appendable:
            return True

        new_header, header_size = hi.new_header, hi.header_size
        data_length = hi.data_length

        # Set buffer size to 16 MiB to hide the Python loop overhead, see
        # https://github.com/numpy/numpy/blob/main/numpy/lib/format.py
        buffersize = min(16 * 1024 ** 2, data_length)
        buffer_count = int(ceil(data_length / buffersize))

        if inplace:
            for i in reversed(range(buffer_count)):
                offset = i * buffersize
                fp.seek(header_size + offset, SEEK_SET)
                content = fp.read(buffersize)
                fp.seek(new_header_size + offset, SEEK_SET)
                fp.write(content)

            fp.seek(0, SEEK_SET)
            fp.write(new_header)

            return True

        dirname, basename = os.path.split(fp.name)

        fp2 = open(tempfile.NamedTemporaryFile(
            prefix=basename, dir=dirname, delete=False
        ).name, 'wb+')
        fp2.write(new_header)

        fp.seek(header_size, SEEK_SET)
        for _ in range(buffer_count):
            fp2.write(fp.read(buffersize))

    fp2.close()
    os.replace(fp2.name, fp.name)

    return True

def recover(filename, zerofill_incomplete=False):
    with open(filename, mode="rb+") as fp:
        hi = _HeaderInfo(fp)
        shape, fortran_order, dtype = hi.shape, hi.fortran_order, hi.dtype
        header_size, data_length = hi.header_size, hi.data_length

        if not hi.needs_recovery:
            return True

        if not hi.is_appendable:
            msg = "header not appendable, please call ensure_appendable first"
            raise ValueError(msg)

        append_axis_itemsize = prod(
            shape[slice(None, None, -1 if fortran_order else 1)][1:]
        ) * dtype.itemsize

        trailing_bytes = data_length % append_axis_itemsize

        if trailing_bytes != 0:
            if zerofill_incomplete is True:
                zero_bytes_to_append = append_axis_itemsize - trailing_bytes
                fp.write(b'\0'*(zero_bytes_to_append))
                data_length += zero_bytes_to_append
            else:
                fp.truncate(header_size + data_length - trailing_bytes)
                data_length -= trailing_bytes

        new_shape = list(shape)
        new_shape[-1 if fortran_order else 0] = \
            data_length // append_axis_itemsize

        fp.seek(0, SEEK_SET)
        _write_array_header(fp, {
            "shape": tuple(new_shape),
            "fortran_order": fortran_order,
            "descr": format.dtype_to_descr(dtype)
        }, header_len=header_size)

    return True

class NpyAppendArray:
    fp = None
    __lock, __is_init, __header_length = threading.Lock(), False, None

    def __init__(
        self, filename, delete_if_exists=False,
        rewrite_header_on_append=True
    ):
        self.filename = filename
        self.__rewrite_header_on_append = rewrite_header_on_append

        if os.path.exists(filename):
            if delete_if_exists:
                os.unlink(filename)
            else:
                self.__init_from_file()

    def __init_from_file(self):
        fp = open(self.filename, "rb+")
        self.fp = fp

        hi = _HeaderInfo(fp)
        self.shape, self.fortran_order, self.dtype, self.__header_length = (
            hi.shape, hi.fortran_order, hi.dtype, hi.header_size
        )

        if self.dtype.hasobject:
            raise ValueError("Object arrays cannot be appended to")

        if not hi.is_appendable:
            msg = (
                "header of {} not appendable, please call "
                "npy_append_array.ensure_appendable"
            ).format(self.filename)
            raise ValueError(msg)

        if hi.needs_recovery:
            msg = (
                "cannot append to {}: file needs recovery, please call " 
                "npy_append_array.recover"
            ).format(self.filename)
            raise ValueError(msg)

        self.__is_init = True

    def __write_array_header(self):
        fp = self.fp
        fp.seek(0, SEEK_SET)

        _write_array_header(fp, {
            "shape": self.shape,
            "fortran_order": self.fortran_order,
            "descr": format.dtype_to_descr(self.dtype)
        }, header_len = self.__header_length)

    def update_header(self):
        with self.__lock:
            self.__write_array_header()

    def append(self, arr):
        with self.__lock:
            if not self.__is_init:
                with open(self.filename, 'wb') as fp:
                    write_array(fp, arr)
                self.__init_from_file()
                return

            shape = self.shape
            fortran_order = self.fortran_order
            fortran_coeff = -1 if fortran_order else 1

            if shape[::fortran_coeff][1:][::fortran_coeff] != \
            arr.shape[::fortran_coeff][1:][::fortran_coeff]:
                msg = "array shapes can only differ on append axis: " \
                "0 if C order or -1 if fortran order"

                raise ValueError(msg)

            self.fp.seek(0, SEEK_END)

            arr.astype(self.dtype, copy=False).flatten(
                order='F' if self.fortran_order else 'C'
            ).tofile(self.fp)

            self.shape = (*shape[:-1], shape[-1] + arr.shape[-1]) \
                if fortran_order else (shape[0] + arr.shape[0], *shape[1:])

            if self.__rewrite_header_on_append:
                self.__write_array_header()

    def close(self):
        with self.__lock:
            if self.__is_init:
                if not self.__rewrite_header_on_append:
                    self.__write_array_header()

                self.fp.close()

                self.__is_init = False

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()
