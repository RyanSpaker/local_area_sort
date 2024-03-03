use std::cmp::Ordering;
use macroquad::{prelude::*, color};

pub trait Distance{
    //Distance from self to other, in the range -1 to 1
    fn dist(&self, other: &Self) -> f32;
    fn avg(&self, other: &Self) -> Self;
    fn flip(&self, other: &Self) -> Self;
}
impl Distance for f32{
    fn dist(&self, other: &f32) -> f32{other - self}
    fn avg(&self, other: &f32) -> f32{(self+other)*0.5}
    fn flip(&self, other: &Self) -> Self {
        2.0*other - self
    }
}

#[derive(PartialEq, Clone, Copy)]
struct Hsl {
    h: f32,
    s: f32,
    l: f32,
    comparison_item: usize
}
#[allow(dead_code)]
impl Hsl{
    pub fn new(vals: (f32, f32, f32), comp: usize) -> Hsl{
        Hsl{h: vals.0, s: vals.1, l: vals.2, comparison_item: comp}
    }
    pub fn to_color(&self) -> Color{
        color::hsl_to_rgb(self.h, self.s, self.l)
    }
}
impl PartialOrd for Hsl{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.comparison_item {
            0 => { return self.h.partial_cmp(&other.h); },
            1 => { return self.s.partial_cmp(&other.s); },
            _ => { return self.l.partial_cmp(&other.l); }
        }
    }
}
impl Distance for Hsl{
    fn dist(&self, other:  &Hsl) -> f32{
        match self.comparison_item {
            0 => { (other.h - self.h + 0.5).rem_euclid(1.0) - 0.5 },
            1 => { other.s - self.s },
            _ => { other.l - self.l }
        }
    }
    fn avg(&self, other: &Hsl) -> Hsl{
        Hsl{
            h: (((other.h - self.h + 0.5).rem_euclid(1.0) - 0.5)*0.5 + self.h).rem_euclid(1.0),
            s: (self.s + other.s)*0.5, 
            l: (self.l + other.l)*0.5, 
            comparison_item: self.comparison_item
        }
    }
    fn flip(&self, _other: &Self) -> Self {
        return self.clone();
    }
}
