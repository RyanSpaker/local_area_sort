pub mod drawing;
pub mod types;

use std::f32::consts::PI;
use std::iter;
use std::ops::Sub;
use itertools::Itertools;
use macroquad::prelude::*;
use ::rand::seq::SliceRandom;
use ::rand::{thread_rng, Rng, rngs::StdRng, SeedableRng};

use crate::drawing::*;
use crate::types::Distance;

const SIZE: Vec2 = Vec2{x: 500.0, y: 500.0};
const NUM_POINTS: usize = 10;
const PERIOD: f32 = 50.0;

#[macroquad::main("LocalAreaSort")]
async fn main() {
    request_new_screen_size(SIZE.x, SIZE.y);
    map().await;
}

pub async fn graphs(){
    //gets rng from seed
    let seed: u64 = thread_rng().gen_range(0..u64::MAX);
    let mut rng: StdRng = StdRng::seed_from_u64(seed);
    println!("seed: {}", seed);
    //sets points as sine curve with time axis error
    let points: Vec<f32> = (0..NUM_POINTS).into_iter().map(|i| {
        ((i as f32 / PERIOD * 2.0 * PI).sin()/2.0 + 0.5,
        i as f32 + rng.gen_range(-5.0..5.0))
    }).sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|a| a.0).collect();
    let mut iterations: usize = 1;
    loop{
        let (sorted, delta, residuals) = sort_points(&points, iterations);
        clear_background(BLACK);
        draw_arrays_delta(&points, &sorted, &delta, Vec2::ZERO, SIZE*Vec2::new(1.0, 0.7), 0.0, 1.0);
        draw_array(&residuals, SIZE*Vec2::new(1.0, 0.3), SIZE*Vec2::new(0.0, 0.7), -1.0, 1.0, GREEN, RED, true);
        if is_key_pressed(KeyCode::Up) {
            iterations += 1;
            println!("{}", iterations);
        }
        if is_key_pressed(KeyCode::Down) {
            if iterations > 0{iterations -= 1;}
            println!("{}", iterations);
        }

        next_frame().await;
    }
}

pub async fn map(){
    //gets rng from seed
    let seed: u64 = thread_rng().gen_range(0..u64::MAX);
    let mut rng: StdRng = StdRng::seed_from_u64(seed);
    println!("seed: {}", seed);

    let mut data = (0..NUM_POINTS).into_iter().map(|val| val as f32).collect::<Vec<f32>>();
    data.shuffle(&mut rng);
    let mut data: Vec<(usize, f32)> = data.into_iter().enumerate().collect();
    println!("{:?}", data);
    data.shuffle(&mut rng);
    // error, sortedness
    let mut errors = vec![];
    let mut min = 10000.0;
    let mut max = 0.0;
    for _ in 0..10000{
        data.shuffle(&mut rng);
        let cur_error = calculate_sortedness(&data);
        if cur_error > max {max = cur_error;}
        if cur_error < min {min = cur_error;}
        errors.push(cur_error);
    }
    println!("{} {}", min, max);

    loop{
        clear_background(BLACK);
        for val in errors.iter(){
            draw_circle(val*SIZE.x, SIZE.y - SIZE.y*0.5 + rng.gen_range(-20.0..20.0), 1.0, RED);
        }
        next_frame().await;
    }
}

/// Calculates the mean squared error of the data, assuming the original was a vector of 0..data.len(). input is usually useful as the index array
pub fn calculate_error(data: &Vec<(usize, f32)>) -> f32{
    data.iter().zip((0..data.len()).into_iter()).map(|((val, _), original)| {
        (*val as f32 - original as f32).abs().powi(2)
    }).sum::<f32>() / data.len() as f32
}
/// caluclate the sortedness of the data
pub fn calculate_sortedness(data: &Vec<(usize, f32)>) -> f32{
    let mut old = data[0].1;
    let mut total_peaks = 0;
    for i in 1..data.len()-1{
        if (data[i].1 > old && data[i].1 > data[i+1].1) || (data[i].1 < old && data[i].1 < data[i+1].1) {total_peaks += 1;}
        old = data[i].1;
    }
    total_peaks as f32 / (data.len()-2) as f32
}


pub fn smooth_sort(data: &Vec<f32>, smooth_iterations: usize, sort_iterations: usize) -> (Vec<f32>, Vec<f32>, Vec<usize>){
    let mut sorted_data = data.clone();
    let mut delta: Vec<usize> = (0..sorted_data.len()).collect();
    for i in 0..sort_iterations{
        let (smoothed_data, residuals) = smooth_curve(&sorted_data, smooth_iterations);
        let mut affected_ranges: Vec<i32> = vec![0; sorted_data.len()];
        for i in 0..sorted_data.len(){
            if residuals[i] > 0.05 && i < residuals.len()-1 && residuals[i+1] < -0.05{
                let affect = (residuals[i] + residuals[i+1])*0.5;
                affected_ranges[i] = 1;
            }else if residuals[i] < -0.05 && i > 0 && sorted_data[i-1] > 0.05{
                let affect = (residuals[i] + residuals[i-1])*0.5;
                affected_ranges[i] = -1;
            }
        }
        'outer: for i in 0..sorted_data.len(){
            let mut j = (i as i32 - 4).max(0);
            while j < i as i32{
                if affected_ranges[j as usize] + j >= (i as i32).min(i as i32+affected_ranges[i]) && residuals[j as usize].abs() > residuals[i].abs(){continue 'outer;}
                j+=1;
            }
            let mut j = (i as i32 + 4).min(sorted_data.len() as i32-1);
            while j > i as i32{
                if affected_ranges[j as usize] + j <= (i as i32).max(i as i32+affected_ranges[i]) && residuals[j as usize].abs() > residuals[i].abs(){continue 'outer;}
                j-=1;
            }

            //ok we are def good to swap now
            let item = sorted_data.remove(i);
            sorted_data.insert((i as i32+affected_ranges[i]) as usize, item);
            let d = delta.remove(i);
            delta.insert((i as i32+affected_ranges[i]) as usize, d)
        }
    }
    let mut delta: Vec<(usize, usize)> = delta.into_iter().enumerate().collect();
    delta.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let delta: Vec<usize> = delta.into_iter().map(|val| val.0).collect();
    let (_, residuals) = smooth_curve(&sorted_data, smooth_iterations);
    return (sorted_data, residuals, delta);
}
pub fn smooth_curve(data: &Vec<f32>, iterations: usize) -> (Vec<f32>, Vec<f32>){
    let mut smoothed_data = data.clone();
    smoothed_data.insert(0, smoothed_data[1].flip(&smoothed_data[0]));
    smoothed_data.insert(0, smoothed_data[1].flip(&smoothed_data[0]));
    smoothed_data.push(smoothed_data[smoothed_data.len()-2].flip(&smoothed_data[smoothed_data.len()-1]));
    smoothed_data.push(smoothed_data[smoothed_data.len()-2].flip(&smoothed_data[smoothed_data.len()-1]));
    for _ in 0..iterations{
        let mut next_iteration = smoothed_data.clone();
        for i in 2..smoothed_data.len()-2{
            next_iteration[i] = (smoothed_data[i-2] + smoothed_data[i-1] + smoothed_data[i] + smoothed_data[i+1] + smoothed_data[i+2])*0.2;
        }
        smoothed_data = next_iteration;
    }
    let mut residuals = data.clone();
    for i in 0..residuals.len(){
        residuals[i] = residuals[i]-smoothed_data[i+2];
    }
    return (smoothed_data, residuals);
}
pub fn sort_points(data: &Vec<f32>, iterations: usize) -> (Vec<f32>, Vec<usize>, Vec<f32>){
    let mut sorted = data.clone();
    let mut delta: Vec<usize> = (0..sorted.len()).collect();
    let mut residuals = calculate_residuals_v3(&sorted);
    for _ in 0..iterations{
        residuals = calculate_residuals_v3(&sorted);
        let mut affected_ranges: Vec<i32> = vec![0; sorted.len()];
        for i in 0..sorted.len(){
            if residuals[i] > 0.05 && i < residuals.len()-1 && residuals[i+1] < -0.05{
                let affect = (residuals[i] + residuals[i+1])*0.5;
                affected_ranges[i] = 1;
            }else if residuals[i] < -0.05 && i > 0 && sorted[i-1] > 0.05{
                let affect = (residuals[i] + residuals[i-1])*0.5;
                affected_ranges[i] = -1;
            }
        }
        'outer: for i in 0..sorted.len(){
            let mut j = (i as i32 - 4).max(0);
            while j < i as i32{
                if affected_ranges[j as usize] + j >= (i as i32).min(i as i32+affected_ranges[i]) && residuals[j as usize].abs() > residuals[i].abs(){continue 'outer;}
                j+=1;
            }
            let mut j = (i as i32 + 4).min(sorted.len() as i32-1);
            while j > i as i32{
                if affected_ranges[j as usize] + j <= (i as i32).max(i as i32+affected_ranges[i]) && residuals[j as usize].abs() > residuals[i].abs(){continue 'outer;}
                j-=1;
            }

            //ok we are def good to swap now
            let item = sorted.remove(i);
            sorted.insert((i as i32+affected_ranges[i]) as usize, item);
            let d = delta.remove(i);
            delta.insert((i as i32+affected_ranges[i]) as usize, d)
        }
    }
    let mut delta: Vec<(usize, usize)> = delta.into_iter().enumerate().collect();
    delta.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let delta: Vec<usize> = delta.into_iter().map(|val| val.0).collect();
    residuals = calculate_residuals_v3(&sorted);
    return (sorted, delta, residuals);
}
pub fn calculate_residuals(data: &Vec<f32>) -> Vec<f32>{
    let mut new_data: Vec<f32> = vec![];
    data.windows(3).for_each(|window| {
        new_data.push(window[1].dist(&window[0].avg(&window[2]))*(window[0]-window[2]).signum());
    });
    new_data.insert(0, data[0].dist(&data[2].flip(&data[1])));
    new_data.push(data[data.len()-1].dist(&data[data.len()-3].flip(&data[data.len()-2])));
    return new_data;
}
pub fn calculate_residuals_v2(data: &Vec<f32>) -> Vec<f32>{
    let mut new_data: Vec<f32> = vec![];
    let mut old_data = data.clone();
    old_data.insert(0, old_data[1].flip(&old_data[0]));
    old_data.insert(0, old_data[1].flip(&old_data[0]));
    old_data.push(old_data[old_data.len()-2].flip(&old_data[old_data.len()-1]));
    old_data.push(old_data[old_data.len()-2].flip(&old_data[old_data.len()-1]));
    old_data.windows(5).for_each(|window| {
        let a = 
            area(Vec2::new(1.0, window[1]), Vec2::new(2.0, window[2]), Vec2::new(3.0, window[3])) + 
            area(Vec2::new(2.0, window[2]), Vec2::new(3.0, window[3]), Vec2::new(4.0, window[4])) - 
            area(Vec2::new(1.0, window[1]), Vec2::new(2.0, window[3]), Vec2::new(3.0, window[2])) - 
            area(Vec2::new(3.0, window[2]), Vec2::new(2.0, window[3]), Vec2::new(4.0, window[4]));
        let b = 
            area(Vec2::new(0.0, window[0]), Vec2::new(1.0, window[1]), Vec2::new(2.0, window[2])) + 
            area(Vec2::new(1.0, window[1]), Vec2::new(2.0, window[2]), Vec2::new(3.0, window[3])) - 
            area(Vec2::new(0.0, window[0]), Vec2::new(2.0, window[1]), Vec2::new(1.0, window[2])) - 
            area(Vec2::new(2.0, window[1]), Vec2::new(1.0, window[2]), Vec2::new(3.0, window[3]));
        new_data.push((a-b)*window[2].dist(&window[1].avg(&window[3])).abs()*10.0);
    });
    return new_data;
}
pub fn calculate_residuals_v3(data: &Vec<f32>) -> Vec<f32>{
    let mut new_data: Vec<f32> = vec![];
    let mut old_data = data.clone();
    old_data.insert(0, old_data[1].flip(&old_data[0]));
    old_data.insert(0, old_data[1].flip(&old_data[0]));
    old_data.push(old_data[old_data.len()-2].flip(&old_data[old_data.len()-1]));
    old_data.push(old_data[old_data.len()-2].flip(&old_data[old_data.len()-1]));
    old_data.windows(5).for_each(|window| {
        let left = (2f32*window[2] - window[1] - window[3]).abs() + (2f32*window[1] - window[0] - window[2]).abs();
        let right = (2f32*window[2] - window[1] - window[3]).abs() + (2f32*window[3] - window[2] - window[4]).abs();
        let left_fix_resid = (2f32*window[1] - window[2] - window[3]).abs() + (2f32*window[2] - window[0] - window[1]).abs();
        let right_fix_resid = (2f32*window[3] - window[1] - window[2]).abs() + (2f32*window[2] - window[3] - window[4]).abs();
        let left_benefit = left - left_fix_resid;
        let right_benefit = right - right_fix_resid;
        if left_benefit > 0.0 && left_benefit > right_benefit{
            new_data.push(-1.0*left_benefit);
        }else if right_benefit > 0.0{
            new_data.push(right_benefit);
        }else{
            new_data.push(0.0);
        }
    });
    return new_data;
}
pub fn area(p1: Vec2, p2: Vec2, p3: Vec2) -> f32{
    (p1.x*p2.y + p1.y*p3.x + p2.x*p3.y - p1.x*p3.y - p1.y*p2.x - p2.y*p3.x).abs()*0.5
}

/*
async fn test(){
    request_new_screen_size(SIZE.x, SIZE.y);

    let mut kernel_size: usize = 1;

    let seed: u64 = thread_rng().gen_range(0..u64::MAX);
    let mut rng: StdRng = StdRng::seed_from_u64(seed);
    println!("seed: {}", seed);

    let points: Vec<f32> = (0..NUM_POINTS).into_iter().map(|i| {
        ((i as f32 / PERIOD * 2.0 * PI).sin()/2.0 + 0.5,
        i as f32 + rng.gen_range(-5.0..5.0))
    }).sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|a| a.0).collect();

    let error: Vec<f32> = calculate_error(&points);

    let mut int_error: Vec<f32> = integrate_error(&error, kernel_size);

    let slope_error: Vec<f32> = calc_slope_times_error(&points);

    loop{
        clear_background(BLACK);

        if is_key_pressed(KeyCode::Up){
            kernel_size += 1;
            println!("Kernel Size: {}", kernel_size);
            int_error = integrate_error(&error, kernel_size);
        }

        if is_key_pressed(KeyCode::Down){
            if kernel_size > 0 {
                kernel_size -= 1;
            }
            println!("Kernel Size: {}", kernel_size);
            int_error = integrate_error(&error, kernel_size);
        }

        draw_graphs2(Vec2::ZERO, Vec2{x: SIZE.x, y: SIZE.y}, &points, &slope_error, &error, &int_error);

        next_frame().await;
    }
}
async fn old(){
    request_new_screen_size(SIZE.x, SIZE.y);
    
    let mut lerp_factor: f32 = 0.0;
    let mut dist_pow: f32 = 1.0;
    let mut kernel_size: usize = 6;
    
    /*
    let img = Texture2D::from_file_with_format(include_bytes!("../img.png"), None);
    img.set_filter(FilterMode::Nearest);
    let mut sorted = Texture2D::from_image(&img.get_texture_data());
    */

    let seed: u64 = thread_rng().gen_range(0..u64::MAX);
    let mut rng: StdRng = StdRng::seed_from_u64(seed);
    println!("seed: {}", seed);

    let points: Vec<f32> = (0..NUM_POINTS).into_iter().map(|i| {
        ((i as f32 / PERIOD * 2.0 * PI).sin()/2.0 + 0.5,
        i as f32 + rng.gen_range(-5.0..5.0))
    }).sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|a| a.0).collect();

    let mut sorted: Vec<usize> = vec![];
    
    
    let mut needs_update: bool = true;

    loop{
        if needs_update{
            
            sorted = sort(
                &points,
                kernel_size,
                dist_pow,
                lerp_factor
            );
            
            /*
            sort_image(&img, &mut sorted, lerp_factor, kernel_size, dist_pow);
            needs_update = false;
            */
        }
        if is_key_pressed(KeyCode::Up) {
            lerp_factor = 1_f32.min(lerp_factor+0.05);
            println!("lerp factor: {}", lerp_factor);
            needs_update = true;
        }
        if is_key_pressed(KeyCode::Down) {
            lerp_factor = 0_f32.max(lerp_factor-0.05);
            println!("lerp factor: {}", lerp_factor);
            needs_update = true;
        }
        if is_key_pressed(KeyCode::O) {
            dist_pow = 10_f32.min(dist_pow + 0.25);
            println!("dist pow: {}", dist_pow);
            needs_update = true;
        }
        if is_key_pressed(KeyCode::L) {
            dist_pow = 0.25_f32.max(dist_pow - 0.25);
            println!("dist pow: {}", dist_pow);
            needs_update = true;
        }
        if is_key_pressed(KeyCode::I) {
            kernel_size = 100.min(kernel_size + 1);
            println!("kernel size: {}", kernel_size);
            needs_update = true;
        }
        if is_key_pressed(KeyCode::K) {
            kernel_size = 2.max(kernel_size - 1);
            println!("kernel size: {}", kernel_size);
            needs_update = true;
        }
        clear_background(BLACK);
        
        //draw_texture(sorted, 0.0, 0.0, WHITE);

        draw_graphs(Vec2::ZERO, Vec2{x: SIZE.x, y: SIZE.y}, &points, &sorted);

        next_frame().await;
    }
}

pub fn sort_image(img: &Texture2D, sorted: &mut Texture2D, lerp_factor: f32, kernel_size: usize, dist_pow: f32){
    let mut data = img.get_texture_data();
    let width: usize = data.width();
    data.get_image_data_mut().chunks_mut(width).for_each(|chunk| {
        let unsorted: Vec<Hsl> = chunk.iter().map(|v| 
            Hsl::new(color::rgb_to_hsl(color::Color::from_rgba(v[0], v[1], v[2], v[3])), 0)
        ).collect();
        let sorted: Vec<usize> = sort(
            &unsorted,
            kernel_size, 
            dist_pow, 
            lerp_factor
        );
        chunk.iter_mut().enumerate().for_each(|(i, v)| {
            let color = unsorted[sorted[i]].to_color();
            v[0] = (color.r * 255.0) as u8;
            v[1] = (color.g * 255.0) as u8;
            v[2] = (color.b * 255.0) as u8;
        });
    });
    sorted.update(&data);
}
pub fn calculate_error<T: Distance>(points: &Vec<T>) -> Vec<f32>{
    let mut errors: Vec<f32> = points.windows(3).map(|w| {
        w[0].avg(&w[2]).dist(&w[1])
    }).collect();
    errors.insert(0, 0.0);
    errors.push(0.0);
    return errors
}
pub fn calc_slope_times_error(points: &Vec<f32>) -> Vec<f32>{
    let mut errors: Vec<f32> = points.windows(3).map(|w| {
        (w[2] - w[0])/2.0 * 
        (w[1] - (w[0]+w[2])/2.0) * 10.0
    }).collect();
    errors.insert(0, 0.0);
    errors.push(0.0);
    return errors
}
pub fn integrate_error(error: &Vec<f32>, kernel_size: usize) -> Vec<f32>{
    let mut int: Vec<f32> = vec![];
    for i in 0..error.len(){
        let mut sum: f32 = 0.0;
        let mut cursor: i32 = i as i32 - kernel_size as i32;
        while cursor < (i+kernel_size+1) as i32 {
            if cursor >= 0 && cursor < error.len() as i32{sum += error[cursor as usize];}
            cursor += 1;
        }
        sum /= 2.0*kernel_size as f32 + 1.0;
        int.push(sum*50.0);
    }
    return int;
}
pub fn sort<T: PartialOrd + Distance>(list: &Vec<T>, kernel_size: usize, dist_pow: f32, lerp_factor: f32) -> Vec<usize>{
    let mut sorted: Vec<usize> = vec![0];
    let step: f32 = 1.0/(kernel_size as f32 - 1.0);
    let mut buffer: Vec<(usize, f32, f32)> = (0..kernel_size+1)
        .into_iter()
        .map(|i| (i, 1.0-step*(i as f32-1.0), 0.0))
        .sorted_by(|a, b| list[a.0].partial_cmp(&list[b.0]).unwrap())
        .collect();
    let mut old: usize = buffer.iter().find_position(|v| v.0.eq(&0)).unwrap().0;
    let mut rule: i32 = list[0].dist(&list[1]).signum() as i32;
    let mut best: usize;

    for i in 1..list.len(){
        //Assign Ranking Scores
        {
            let mut cursor: usize = (old as i32 + rule).rem_euclid(buffer.len() as i32) as usize;
            let mut score: f32 = 0.0;
            //step one, increase forward
            while list[buffer[old].0].dist(&list[buffer[cursor as usize].0])*rule as f32 >= 0.0 && score < buffer.len() as f32{
                buffer[cursor].2 = score;
                score += 1.0;
                cursor = (cursor as i32 + rule).rem_euclid(buffer.len() as i32) as usize;
            }
            //step two, increase backwards
            cursor = (old as i32 - rule).rem_euclid(buffer.len() as i32) as usize;
            while score < buffer.len() as f32 - 1.0 {
                buffer[cursor].2 = score;
                score += 1.0;
                cursor = (cursor as i32 - rule).rem_euclid(buffer.len() as i32) as usize;
            }
        }
        //remove old value from list
        buffer.remove(old);
        //find best score
        best = buffer.iter()
            .map(|v| v.1.powf(dist_pow)*lerp_factor + (1.0-lerp_factor)*(1.0 - v.2*step))
            .position_max_by(|a,b| a.partial_cmp(&b).unwrap()).unwrap();
        //put best score in final
        sorted.push(buffer[best].0);
        //update distance
        buffer.iter_mut().for_each(|v| v.1 += step);
        //insert new value
        old = best;
        if i+kernel_size < list.len(){
            let mut j: usize = 0;
            while j < buffer.len() && list[i+kernel_size].partial_cmp(&list[buffer[j].0]).unwrap()==Ordering::Greater {j+=1;}
            buffer.insert(j, (i+kernel_size, 0.0, 0.0));
            if j <= old {old += 1;}
        }
        //find new rule
        rule = list[sorted[sorted.len()-2]].dist(&list[buffer[old].0]).signum() as i32;
    }
    return sorted;
}
pub fn calculate_sortedness(values: &Vec<u32>) -> f32{
    values.windows(3).map(|w| {
        if w[0] >= w[1] && w[1] >= w[2] {return 1.0;}
        if w[0] <= w[1] && w[1] <= w[2] {return 1.0;}
        return 0.0;
    }).sum::<f32>() / (values.len() as f32 - 2_f32)
}
pub fn draw_graphs(origin: Vec2, size: Vec2, points: &Vec<f32>, sorted: &Vec<usize>){
    let x_scale: f32 = size.x/(points.len() as f32 - 1.0);
    let mut y_scale: f32 = size.y*0.4;
    points.windows(2).enumerate().for_each(|(i, w)| {
        draw_line(
            i as f32 * x_scale + origin.x, 
            w[0]*y_scale + origin.y,
            (i as f32 + 1.0)*x_scale + origin.x,
            w[1]*y_scale + origin.y,
            1.0,
            GREEN
        );
        draw_circle(
            i as f32 * x_scale + origin.x, 
            w[0]*y_scale + origin.y,
            3.0,
            RED
        );
    });
    draw_circle(
        (points.len() as f32-1.0) * x_scale + origin.x, 
        points.last().unwrap()*y_scale + origin.y,
        3.0,
        RED
    );
    let mut y_origin: f32 = origin.y+size.y*0.6;
    sorted.windows(2).enumerate().for_each(|(i, w)| {
        draw_line(
            i as f32 * x_scale + origin.x, 
            points[w[0]]*y_scale + y_origin,
            (i as f32 + 1.0)*x_scale + origin.x,
            points[w[1]]*y_scale + y_origin,
            1.0,
            GREEN
        );
        draw_circle(
            i as f32 * x_scale + origin.x, 
            points[w[0]]*y_scale + y_origin,
            3.0,
            RED
        );
    });
    /*
    draw_circle(
        (sorted.len() as f32-1.0) * x_scale + origin.x, 
        points[*sorted.last().unwrap()]*y_scale + y_origin,
        3.0,
        RED
    );*/
    y_origin = origin.y + size.y*0.4;
    y_scale = size.y*0.2;
    sorted.iter().enumerate().for_each(|(i, v)| {
        draw_line(
            *v as f32 * x_scale + origin.x,
            y_origin,
            i as f32 * x_scale + origin.x,
            y_origin + y_scale,
            1.0,
            GREEN
        );
    });
}
pub fn draw_graphs2(origin: Vec2, size: Vec2, points: &Vec<f32>, error: &Vec<f32>, int_error: &Vec<f32>, slope_error: &Vec<f32>){
    let x_scale: f32 = size.x/(points.len() as f32 - 1.0);
    let mut y_scale: f32 = size.y*0.2;
    points.windows(2).enumerate().for_each(|(i, w)| {
        draw_line(
            i as f32 * x_scale + origin.x, 
            SIZE.y - (w[0]*y_scale + origin.y),
            (i as f32 + 1.0)*x_scale + origin.x,
            SIZE.y - (w[1]*y_scale + origin.y),
            1.0,
            GREEN
        );
        draw_circle(
            i as f32 * x_scale + origin.x, 
            SIZE.y - (w[0]*y_scale + origin.y),
            3.0,
            RED
        );
    });
    draw_circle(
        (points.len() as f32-1.0) * x_scale + origin.x, 
        SIZE.y - (points.last().unwrap()*y_scale + origin.y),
        3.0,
        RED
    );
    let mut y_origin: f32 = origin.y+size.y*0.35;
    y_scale = size.y*0.15;
    draw_line(origin.x, SIZE.y - (y_origin), origin.x+size.x, SIZE.y - (y_origin), 1.0, BLUE);
    error.windows(2).enumerate().for_each(|(i, w)| {
        draw_line(
            i as f32 * x_scale + origin.x, 
            SIZE.y - (w[0]*y_scale + y_origin),
            (i as f32 + 1.0)*x_scale + origin.x,
            SIZE.y - (w[1]*y_scale + y_origin),
            1.0,
            GREEN
        );
        draw_circle(
            i as f32 * x_scale + origin.x, 
            SIZE.y - (w[0]*y_scale + y_origin),
            3.0,
            RED
        );
    });
    y_origin = origin.y+size.y*0.65;
    y_scale = size.y*0.15;
    draw_line(origin.x, SIZE.y - (y_origin), origin.x+size.x, SIZE.y - (y_origin), 1.0, BLUE);
    int_error.windows(2).enumerate().for_each(|(i, w)| {
        draw_line(
            i as f32 * x_scale + origin.x, 
            SIZE.y - (w[0]*y_scale + y_origin),
            (i as f32 + 1.0)*x_scale + origin.x,
            SIZE.y - (w[1]*y_scale + y_origin),
            1.0,
            GREEN
        );
        draw_circle(
            i as f32 * x_scale + origin.x, 
            SIZE.y - (w[0]*y_scale + y_origin),
            3.0,
            RED
        );
    });
    y_origin = origin.y+size.y*0.9;
    y_scale = size.y*0.1;
    draw_line(origin.x, SIZE.y - (y_origin), origin.x+size.x, SIZE.y - (y_origin), 1.0, BLUE);
    slope_error.windows(2).enumerate().for_each(|(i, w)| {
        draw_line(
            i as f32 * x_scale + origin.x, 
            SIZE.y - (w[0]*y_scale + y_origin),
            (i as f32 + 1.0)*x_scale + origin.x,
            SIZE.y - (w[1]*y_scale + y_origin),
            1.0,
            GREEN
        );
        draw_circle(
            i as f32 * x_scale + origin.x, 
            SIZE.y - (w[0]*y_scale + y_origin),
            3.0,
            RED
        );
    });
}*/