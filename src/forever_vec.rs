use memmap2::{MmapMut, RemapOptions};
use std::fs::{File, OpenOptions};
use std::path::PathBuf;

pub struct ForeverVec<T> {
    file: File,
    mmap: MmapMut,
    len: usize,
    phantom: std::marker::PhantomData<T>,
}

impl<T> ForeverVec<T> {
    fn new(path: PathBuf) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        let size_file = file.metadata()?.len();
        let size_t = std::mem::size_of::<T>() as u64;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(ForeverVec {
            file,
            mmap,
            len: (size_file / size_t) as usize,
            phantom: std::marker::PhantomData,
        })
    }
    fn push(&mut self, value: T) {
        self.len += 1;
        let new_byte_len = std::mem::size_of::<T>() as u64 * self.len as u64;
        self.file.set_len(new_byte_len).unwrap();
        let options = RemapOptions::new();
        options.may_move(true);
        unsafe {
            self.mmap.remap(new_byte_len as usize, options);
        };
        let slice =
            unsafe { std::slice::from_raw_parts_mut(self.mmap.as_mut_ptr() as *mut T, self.len) };
        slice[self.len - 1] = value;
        self.mmap.flush().unwrap();
    }
    fn get(&mut self, index: usize) -> Option<&T> {
        let slice =
            unsafe { std::slice::from_raw_parts_mut(self.mmap.as_mut_ptr() as *mut T, self.len) };
        slice.get(index)
    }
    fn set(&mut self, index: usize, value: T) {
        let slice =
            unsafe { std::slice::from_raw_parts_mut(self.mmap.as_mut_ptr() as *mut T, self.len) };
        slice[index] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::remove_file;
    use std::path::PathBuf;
    #[test]
    fn test_forever_vec() {
        let path = PathBuf::from("./test_forever_vec");
        let mut forever_vec = ForeverVec::<u8>::new(path.clone()).unwrap();
        //forever_vec.push(b'h');
        //forever_vec.push(b'e');
        //forever_vec.push(b'l');
        //forever_vec.push(b'l');
        //forever_vec.push(b'o');
        assert_eq!(forever_vec.len, 5);
        assert_eq!(forever_vec.get(0), Some(&b'h'));
    }
}
