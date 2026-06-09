use std::borrow::Cow;
use std::fs::File;
use std::io::prelude::*;
use std::slice;
use std::mem::{transmute, size_of};
use std::ptr;
use std::io::{Write, Result, Error, ErrorKind};

/// object used to extend functionality of File
/// used for reading and writing byte vectors to files
pub trait VecIO {
    fn write_slice_to_file<T>(&mut self, data: &[T]) -> Result<()>;
    fn read_vec_from_file<T: Copy>(&mut self) -> Result<Vec<T>>;
}

impl VecIO for File {
    /// Writes a vector of type T to file as bytes
    fn write_slice_to_file<T>(&mut self, data: &[T]) -> Result<()> {
        unsafe {
            self.write_all(slice::from_raw_parts(transmute::<*const T, *const u8>(data.as_ptr()), data.len() * size_of::<T>()))?;
        }
        Ok(())
    }
    /// Reads a Vector of type T from file
    ///
    /// Reads into a `u8` buffer and copies into a freshly allocated `Vec<T>`.
    /// Reinterpreting the `u8` buffer in place via `from_raw_parts` would be
    /// unsound: the buffer is only `u8`-aligned, and dropping the resulting
    /// `Vec<T>` would deallocate with a layout that never matches the original
    /// `u8` allocation. `T: Copy` rules out types with `Drop`/non-trivial init.
    fn read_vec_from_file<T: Copy>(&mut self) -> Result<Vec<T>> {
        let mut bytes: Vec<u8> = Vec::new();
        self.read_to_end(&mut bytes)?;
        if bytes.len() % size_of::<T>() != 0 {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                format!("read_file() returned a number of bytes ({}) which is not a multiple of size ({})", bytes.len(), size_of::<T>())
            ));
        }
        let length = bytes.len() / size_of::<T>();
        let mut buffer: Vec<T> = Vec::with_capacity(length);
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.as_mut_ptr() as *mut u8, bytes.len());
            buffer.set_len(length);
        }
        Ok(buffer)
    }
}

/// Helper function used to unpack vectors from RustEmbed Assets
///
/// This is data that is embeded in the binary
///
/// [Rust Embed]: https://crates.io/crates/rust-embed
///
/// Copies the asset bytes into a freshly allocated `Vec<T>`. Reinterpreting the
/// asset buffer in place via `from_raw_parts` is unsound: the bytes are only
/// `u8`-aligned (so a `*mut T` may be misaligned), and the resulting `Vec<T>`
/// would later deallocate with a `(len*size_of::<T>(), align_of::<T>())` layout
/// that does not match the original allocation — undefined behavior independent
/// of `T`'s `Drop`. `T: Copy` additionally rules out types with destructors.
pub fn unpack_vec_from_asset<T: Copy>(asset: Option<Cow<'static, [u8]>>) -> Result<Vec<T>> {
    let bytes = match asset {
        Some(data) => data,
        None => {
            return Err(Error::new(
                ErrorKind::NotFound,
                format!("unable to read asset, file not found")
            ));
        }
    };
    if bytes.len() % size_of::<T>() != 0 {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            format!("read_asset() returned a number of bytes which is not a multiple of size ({})", size_of::<T>())
        ));
    }
    let length = bytes.len() / size_of::<T>();
    let mut buffer: Vec<T> = Vec::with_capacity(length);
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.as_mut_ptr() as *mut u8, bytes.len());
        buffer.set_len(length);
    }
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// native-endian byte view of a slice, matching `write_slice_to_file`'s format
    fn as_bytes<T>(data: &[T]) -> Vec<u8> {
        unsafe {
            slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * size_of::<T>()).to_vec()
        }
    }

    // u32 has align 4; round-tripping through a u8 asset must produce correctly
    // aligned, correctly valued data with no allocator-layout UB on drop.
    #[test]
    fn unpack_u32_roundtrip() {
        let values: Vec<u32> = vec![0xDEAD_BEEF, 0x0011_2233, 7, u32::MAX];
        let asset: Option<Cow<'static, [u8]>> = Some(Cow::Owned(as_bytes(&values)));
        let unpacked: Vec<u32> = unpack_vec_from_asset(asset).unwrap();
        assert_eq!(values, unpacked);
        // pointer must satisfy u32 alignment
        assert_eq!(unpacked.as_ptr() as usize % std::mem::align_of::<u32>(), 0);
    }

    #[test]
    fn unpack_rejects_non_multiple() {
        let asset: Option<Cow<'static, [u8]>> = Some(Cow::Owned(vec![1u8, 2, 3]));
        let res: Result<Vec<u32>> = unpack_vec_from_asset(asset);
        assert!(res.is_err());
    }

    #[test]
    fn file_roundtrip_u16() {
        let values: Vec<u16> = vec![1, 2, 3, 0xFFFF, 42];
        let path = std::env::temp_dir().join("read_write_roundtrip_u16.dat");
        {
            let mut f = File::create(&path).unwrap();
            f.write_slice_to_file(&values).unwrap();
        }
        let read_back: Vec<u16> = File::open(&path).unwrap().read_vec_from_file::<u16>().unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(values, read_back);
    }
}
