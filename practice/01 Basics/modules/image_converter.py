import numpy as np
import pandas as pd
import math
import cv2
import imutils
#from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

class Image2TimeSeries:
    """
    Converter from image to time series by angle-based method
        
    Parameters
    ----------
    angle_step: angle step for finding the contour points
    """
    
    def __init__(self, angle_step: int = 10) -> None:
        self.angle_step: int = angle_step


    def _img_preprocess(self, img: np.ndarray) -> np.ndarray:  
        """
        Preprocess the raw image: convert to grayscale, inverse, blur slightly, and threshold it
        
        Parameters
        ----------
        img: raw image
        
        Returns
        -------
        prep_img: image after preprocessing
        """
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        
        _, thresh_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return thresh_img 

    def _get_contour(self, img: np.ndarray) -> np.ndarray:
        """
        Find the largest contour in the preprocessed image

        Parameters
        ----------
        img: preprocessed image
        
        Returns
        -------
        contour: object contour
        """

        contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = [cnt for cnt in contours if cv2.contourArea(cnt) > 500][0]

        return contour


    def _get_center(self, contour: np.ndarray) -> tuple[float, float]:
        """
        Compute the object center

        Parameters
        ----------
        contour: object contour
        
        Returns
        -------
            coordinates of the object center
        """

        M = cv2.moments(contour)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        return (center_x, center_y)


    def _find_nearest_idx(self, array: np.ndarray, value: int) -> int:
        """
        Find index of element that is the nearest to the defined value

        Parameters
        ----------
        array: array of values
        value: defined value
     
        Returns
        -------
        idx: index of element that is the nearest to the defined value
        """

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        
        return idx


    def _get_coordinates_at_angle(self, contour: np.ndarray, center: tuple[float, float], angle: int) -> np.ndarray:
        """
        Find one point on contour that are located at the angle

        Parameters
        ----------
        contour: object contour
        center: object center
        angle: angle
     
        Returns
        -------
            coordinates of one point on the contour
        """

        angles = np.rad2deg(np.arctan2(*(center - contour).T))
        angles = np.where(angles < -90, angles + 450, angles + 90)
        found = np.rint(angles) == angle
        
        if np.any(found):
            return contour[found][0]
        else:
            idx = self._find_nearest_idx(angles, angle)
            return contour[idx]


    def _get_edge_coordinates(self, contour: np.ndarray, center: tuple[float, float]) -> list[np.ndarray]:
        """
        Find points on contour that are located from each other at the angle step

        Parameters
        ----------
        contour: object contour
        center: object center
     
        Returns
        -------
        edge_coordinates: coordinates of the object center
        """

        edge_coordinates = []
        for angle in range(0, 360, self.angle_step):
            pt = self._get_coordinates_at_angle(contour, center, angle)
            if np.any(pt):
                edge_coordinates.append(pt)

        return edge_coordinates


    def _img_show(self, img: np.ndarray, contour: np.ndarray, edge_coordinates: list[np.ndarray], center: tuple[float, float]) -> None:
   
    
        vis_img = img.copy()

       
        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

        
        cv2.drawContours(vis_img_rgb, [contour], -1, (0, 255, 0), 2)
        
  
        for point in edge_coordinates:
            cv2.line(vis_img_rgb, center, tuple(point), (255, 0, 255), 1)

       
        cv2.circle(vis_img_rgb, center, 5, (255, 20, 147), -1)

    
        plt.imshow(vis_img_rgb)
        plt.title("Image with Contours and Rays")
        plt.axis('off')
        plt.show()


    def convert(self, img: np.ndarray, is_visualize: bool = False) -> np.ndarray:
        """
        Convert image to time series by angle-based method

        Parameters
        ----------
        img: input image
        is_visualize: visualize or not image with contours, center and rais from starting center
        
        Returns
        -------
        ts: time series representation
        """

        ts = []

        prep_img = self._img_preprocess(img)

     
        contour = self._get_contour(prep_img)
      
        
        center = self._get_center(contour)
        edge_coordinates = self._get_edge_coordinates(contour.squeeze(), center)

        if (is_visualize):
          
            self._img_show(img.copy(), contour, edge_coordinates, center)

        for coord in edge_coordinates:
            dist = math.sqrt((coord[0] - center[0])**2 + (coord[1] - center[1])**2) 
            ts.append(dist)

        return np.array(ts)
