use std::ops::{Add, Sub, Div};

use macroquad::prelude::*;
use crate::SIZE;

pub fn line(p1: Vec2, p2: Vec2, thickness: f32, color: Color){
    draw_line(
        p1.x, 
        SIZE.y - p1.y, 
        p2.x, 
        SIZE.y - p2.y, 
        thickness, 
        color
    );
}
pub fn circle(p: Vec2, radius: f32, color: Color){
    draw_circle(p.x, SIZE.y-p.y, radius, color);
}
pub fn draw_array<T>(
    data: &Vec<T>, 
    resolution: Vec2, 
    origin: Vec2, 
    min_val: T, max_val: T, 
    line_color: Color,
    point_color: Color,
    draw_center_line: bool
)
where T: PartialOrd + Into<f32> + Add + Sub<Output = T> + Div + Clone + Copy{
    let scale = Vec2::new(resolution.x/(data.len()-1) as f32, resolution.y/(max_val-min_val).into());
    data.windows(2).enumerate().for_each(|(i, w)| {
        line(
            Vec2::new(i as f32, (w[0] - min_val).into())*scale + origin,
            Vec2::new(i as f32 + 1.0, (w[1] - min_val).into())*scale + origin,
            1.0,
            line_color
        );
        circle(
            Vec2::new(i as f32, (w[0] - min_val).into())*scale + origin, 3.0, point_color
        );
    });
    circle(
        Vec2::new(data.len() as f32 - 1.0, (*data.last().unwrap() - min_val).into())*scale + origin, 3.0, point_color
    );
    if draw_center_line{
        line(Vec2::new(0.0, 0.5)*resolution+origin, Vec2::new(1.0, 0.5)*resolution + origin, 1.0, BLUE);
    }
}
pub fn draw_delta(delta: &Vec<usize>, origin: Vec2, resolution: Vec2){
    let scale = Vec2::new(resolution.x/(delta.len()-1) as f32, resolution.y);
    for (i, d) in delta.iter().enumerate(){
        line(
            Vec2::new(i as f32, 0.0)*scale + origin,
            Vec2::new(*d as f32, 1.0)*scale + origin,
            1.0,
            BLUE
        );
    }
}
pub fn draw_arrays_delta<T>(data1: &Vec<T>, data2: &Vec<T>, delta: &Vec<usize>, origin: Vec2, resolution: Vec2, min: T, max: T)
where T: PartialOrd + Into<f32> + Add + Sub<Output = T> + Div + Clone + Copy{
    draw_array(data1, resolution*Vec2::new(1.0, 0.4), 
        origin, min, max, GREEN, RED, false);
    draw_array(data2, resolution*Vec2::new(1.0, 0.4), 
        origin + resolution*Vec2::new(0.0, 0.6), min, max, GREEN, RED, false);
    draw_delta(delta, origin + resolution*Vec2::new(0.0, 0.4), resolution*Vec2::new(1.0, 0.2));
}
pub fn draw_arrays<T>(arrays: &Vec<(&Vec<T>, T, T, bool)>, resolution: Vec2, origin: Vec2)
where T: PartialOrd + Into<f32> + Add + Sub<Output = T> + Div + Clone + Copy{
    let y_step: f32 = resolution.y/arrays.len() as f32;
    for (i, array) in arrays.iter().enumerate(){
        draw_array(&array.0, 
            Vec2::new(resolution.x, y_step), 
            Vec2::new(origin.x, origin.y + y_step*i as f32), array.1, array.2, GREEN, RED, array.3);
    }
}
