#[derive(Debug, Clone, Copy)]
pub struct Bbox {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32, // exclusive
    pub y1: u32, // exclusive
}

#[derive(Debug, Clone)]
pub struct Defect {
    pub pixels: Vec<(u32, u32)>,
    pub bbox: Bbox,
}

impl Defect {
    pub fn max_dim(&self) -> u32 {
        (self.bbox.x1 - self.bbox.x0).max(self.bbox.y1 - self.bbox.y0)
    }
}

/// Connected components over a boolean mask, 4-connectivity (matches
/// scipy.ndimage.label's default used by the training metrics).
pub fn components(mask: &[bool], width: u32, height: u32) -> Vec<Defect> {
    let (w, h) = (width as usize, height as usize);
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    for start in 0..w * h {
        if !mask[start] || seen[start] {
            continue;
        }
        let mut pixels = Vec::new();
        let mut stack = vec![start];
        seen[start] = true;
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        while let Some(i) = stack.pop() {
            let (x, y) = ((i % w) as u32, (i / w) as u32);
            pixels.push((x, y));
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
            let neighbors = [
                (x > 0).then(|| i - 1),
                (x + 1 < width).then(|| i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < height).then(|| i + w),
            ];
            for n in neighbors.into_iter().flatten() {
                if mask[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        out.push(Defect {
            pixels,
            bbox: Bbox { x0, y0, x1, y1 },
        });
    }
    out
}
