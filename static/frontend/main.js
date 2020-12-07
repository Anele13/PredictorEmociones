let model;

const modelURL = 'http://localhost:5000/model';

const preview = document.getElementById("preview");
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const numberOfFiles = document.getElementById("number-of-files");
const fileInput = document.getElementById('file');
const colores = {0:`border border-danger`,
                1:`border border-secondary`,
                2:`border border-success`,
                3:`border border-warning`,
                4:`border border-info`,
                5:`border border-light`,
                6:`border border-dark`,
                7:`border border-white`}

const class_names =  {0:'Enojo',
                      1:'Rechazo',
                      2:'Felicidad',
                      3:'Neutral',
                      4:'Tristeza',
                      5:'Sorpresa'}


const predict = async (modelURL) => {
    visual_pre_prediccion()
    preview.innerHTML = " "
    if (!model) model = await tf.loadLayersModel(modelURL);
    const files = fileInput.files;

    [...files].map(async (img, index) => {
        const data = new FormData();
        data.append('file', img);
        data.append('color',colores[index])
        $($('.image-block')[index]).addClass(colores[index])
        const result = await fetch("/api/prepare",
            {   method: 'POST',
                body: data
            }).then(response => {
                return response.json();
            }).then(result => { 
              if (!result.error){
                for (const [key, value] of Object.entries(result))  {
                    const processedImage = tf.tensor2d(value[0])
                    const prediction = model.predict(tf.reshape(processedImage, shape = [1, processedImage.shape[0],processedImage.shape[0], 1]));
                    const label = prediction.argMax(axis = 1).dataSync()[0];                      
                    const predicciones = prediction.dataSync();
                    const img_source = "data:image/png;base64," + value[1]
                    const color = value[2]
                    mostrar_resultado()
                    renderImageLabel(img, class_names[label], img_source, color, predicciones);
                }
              }
            }).catch(error => {
              mostrar_errores()
            })
    })
};


 
//Renderiza la imagen resultante de la prediccion con los porcentajes de confiabilidad
const renderImageLabel = (img, label, source_img, color, arreglo_conf) => {
    const reader = new FileReader();
    reader.onload = () => {
        var confiabilidad_por_clase = ""
        var mayor_confiabilidad = parseFloat(Math.max( ...arreglo_conf ) * 100).toFixed(2)
        var dict = {}
        
        //Diccionario con el orden original de las clases antes de reordenar por > % de confiabilidad de las predicciones
        arreglo_conf.forEach(function (element, i) {
          dict[element] = class_names[i]
        });

        //Funcion que crea los nodos por clase con su respectiva confiabilidad
        arreglo_conf.sort().reverse().forEach(function (element, i) {
            porcentaje = parseFloat(element * 100).toFixed(2);
            confiabilidad_por_clase += `<div class="col-md-3 text-center">
                                          <div class="c100 p${parseInt(porcentaje)} small orange">
                                              <span>${porcentaje}%</span>
                                              <div class="slice">
                                                  <div class="bar"></div>
                                                  <div class="fill"></div>
                                              </div>
                                          </div>
                                          <h6>${dict[element]}</h6>
                                      </div>`;

        });
        
        //Muestra las imagenes resultado con los nodos de confiabilidad de las demas clases a la derecha
        preview.innerHTML += ` <div class="row border bg-light shadow-lg p-3 mb-5">
                                  <div class="col-md-3">
                                    <div class="card bg-light mb-3 ${color} image-block" style="max-width: 18rem;">
                                      <div class="card-header"><strong>${label} </strong></div>
                                      <div class="card-body">
                                        <div class="image-block">
                                          <img src="${source_img}" id="source"  style="width: 200px; height: 220px;"/>
                                          <p class="card-text">${mayor_confiabilidad}% de confiabilidad </p>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                  <div class="col-md-5">
                                    <br>
                                      <div class="row">
                                          ${confiabilidad_por_clase}
                                      </div>
                                  </div>
                              </div>`;

    };
    reader.readAsDataURL(img);
};


//Un callback para mostrar el tumbnail de la carga de una imagen
$('input[type="file"]').change(function() {
    $('.thumbnail').html('');
    $.each(this.files, function() {
      readURL(this);
    })
});
  

function readURL(file) {
    var reader = new FileReader();
    reader.onload = function(e) {
      $('.thumbnail').append('<img class=image-block src=' + e.target.result + ' style="width: 200px; height: 220px;"/>');
    }
    reader.readAsDataURL(file);
}


//Acomoda la visual para mostrar los resultados de una prediccion
function mostrar_resultado(){
    $('#spinner').hide() //Esconder spinner
    $('#preview').show() //Mostrar row con resultados
    $('#resultado').show()
    $('#predict').prop("disabled", false); //Restaura el boton de predecir
    $('#predict').html(`Predecir`)
}


//Acomoda la visual para enviar una imagen a preprocesar al backend
function visual_pre_prediccion(){
    $('#banner-error').hide()
    $('#preview').hide()
    $('#resultado').hide()
    $('#spinner').show()
    $('#predict').prop("disabled", true);
    $('#predict').html(`<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Trabajando...`)
}


//Si hay errores, acomoda la visual para mostrar errores
function mostrar_errores(){
    $('#banner-error').show() //Muestra el banner con mensaje de error
    $('#spinner').hide();
    $('#predict').prop("disabled", false);
    $('#predict').html(`Predecir`);
}


//Callback para los botones
predictButton.addEventListener("click", () => predict(modelURL));
clearButton.addEventListener("click", () => {preview.innerHTML = ""; $('#preview').hide(); $('#resultado').hide()});
