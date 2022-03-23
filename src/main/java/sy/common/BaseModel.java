package sy.common;

import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import sy.define_exception.NumberParameterException;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/10 22:24
 */
public abstract class BaseModel {
    public List<INDArray> params = null;
    public List<INDArray> grads = null;

    public BaseModel() {}

    public abstract INDArray forward(Object ... xs) throws NumberParameterException;

    public abstract INDArray backward() throws NumberParameterException;

    public abstract INDArray backward(Object ... dout) throws NumberParameterException;

    public void saveParams(String fileName) {
        if(StringUtils.isNotBlank(fileName)) {
            fileName = this.getClass().getName() + ".pkl";
        }

        List<INDArray> newParams = new ArrayList<>();
        for(INDArray indArray : this.params) {
            newParams.add(indArray.castTo(DataType.FLOAT16));
        }

        try(ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(fileName))) {
            os.writeObject(newParams);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadParams(String fileName) throws FileNotFoundException {

        if(StringUtils.isNotBlank(fileName)) {
            fileName = this.getClass().getName() + ".pkl";
        }

       if(Files.notExists(Paths.get(fileName))) {
           throw new FileNotFoundException("File not exists : " + fileName);
       }

       try(ObjectInputStream is = new ObjectInputStream(new FileInputStream(fileName))) {
           this.params = (List<INDArray>) is.readObject();
       } catch (Exception e) {
           e.printStackTrace();
       }

        List<INDArray> newParams = new ArrayList<>();
        for(INDArray indArray : this.params) {
            newParams.add(indArray.castTo(DataType.FLOAT));
        }
        this.params = newParams;
    }

}

