<?php

function png2jpg($originalFile, $outputFile, $quality) {
    $image = imagecreatefrompng($originalFile);
    imagejpeg($image, $outputFile, $quality);
    imagedestroy($image);
}

function _main() {
	
	$imgs = glob('neg/*.png');
	$output_dir = 'neg_jpg';
	$file_names = array();
	foreach($imgs as $img) {
		$file_name = substr(substr($img, strpos($img, '/') + 1), 0, strpos(substr($img, strpos($img, '/')), '.'));
		$file_name = $output_dir.'/'.$file_name.'jpg';
		png2jpg($img, $file_name, 100);
	}
	
}

_main();
